// Starting the search from a design instead of the whole sphere: a saved .txt (or the editor's design) taken back to
// the carving it was built from, and the placement of the mesh it was scored with.
//
// Carving::Build makes, per piece, a body holding
//   - "Sphere": a speck rounded out to the radius,
//   - "Cutter N" (a cut): a cutter the piece is outside of; "Hollow N" (a cut): every hollow,
//   - "Outside cutter N" (a cut feature: "Around", less "Cutter N"): a cutter the piece is inside of,
// and per joint a hinge "Joint N" (closed angle 0). The finishing passes wrap a body's own shapes in a "Core" feature
// and add theirs beside it, or cut named shapes ("... gap", "... leaf room", ...) into the core: those, the hinges'
// hardware and the clearances are what the search doesn't design, so they are left out and done again when it ends.
#include "autodesign.hpp"
#include "util/utils.hpp"
#include <raymath.h>
#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>

namespace Bakugan {
    namespace {
        const char* kTargetTag = "autodesign-target";

        bool StartsWith(const std::string& s, const char* prefix) { return s.rfind(prefix, 0) == 0; }

        // A name's numbers after `prefix`: "Cutter 12" -> { 12 }. Design names are unique, so a second body's copy is
        // "Cutter 12 2", a second sphere "Sphere 2" (Design::UniqueName): { 12, 2 }, { 2 }. False if it isn't the prefix
        // followed by nothing but numbers.
        bool NameNumbers(const std::string& name, const char* prefix, std::vector<int>* numbers) {
            numbers->clear();
            if (!StartsWith(name, prefix)) return false;
            size_t k = strlen(prefix);
            while (k < name.size()) {
                if (name[k] != ' ') return false;
                k++;
                int n = 0;
                size_t digits = 0;
                for (; k < name.size() && isdigit((unsigned char)name[k]) && digits < 9; k++, digits++) n = n * 10 + (name[k] - '0');
                if (digits == 0) return false;
                numbers->push_back(n);
            }
            return true;
        }

        // "Cutter 12" or "Cutter 12 2" -> 12; 0 when it isn't such a name
        int NameId(const std::string& name, const char* prefix) {
            std::vector<int> n;
            return NameNumbers(name, prefix, &n) && (n.size() == 1 || n.size() == 2) && n[0] > 0 ? n[0] : 0;
        }

        // "Sphere", "Sphere 3"; "Core", "Core 2"
        bool IsNamed(const std::string& name, const char* prefix) {
            std::vector<int> n;
            return NameNumbers(name, prefix, &n) && n.size() <= 1;
        }

        // Two superfrustums the same, to `tol` in lengths (a saved file rounds to 9 digits, so copies of one cutter
        // in different bodies come back identical; this only forgives an edit by a hair)
        bool Same(const SuperFrustum& a, const SuperFrustum& b, float tol) {
            auto near = [&](float x, float y, float t) { return fabsf(x - y) <= t; };
            float dot = a.rotation.x * b.rotation.x + a.rotation.y * b.rotation.y + a.rotation.z * b.rotation.z + a.rotation.w * b.rotation.w;
            return Vector3Distance(a.position, b.position) <= tol && fabsf(dot) >= 1.0f - 1e-5f && near(a.size.x, b.size.x, tol) &&
                   near(a.size.y, b.size.y, tol) && near(a.size.z, b.size.z, tol) && near(a.rounding, b.rounding, 1e-4f) &&
                   near(a.taper, b.taper, 1e-4f) && near(a.bulge, b.bulge, 1e-4f) && near(a.dilation, b.dilation, tol);
        }

        bool IsSphere(const Shape& s) { return IsNamed(s.name, "Sphere") && s.op == Op::Union && s.sf.rounding > 0.99f && s.sf.dilation > 0.0f; }
    }

    // ---- the placement, in the file ------------------------------------------------------------------------------------

    std::string SerializeWithTarget(const Design& d, const AutoScore& score) {
        std::string text = d.Serialize();
        if (!score.valid) return text;
        const Placement& t = score.target;
        std::ostringstream s;
        s.precision(9);
        s << "# where the mesh was placed when this was scored (offset, scale, the mesh's size then): the generator carries on from it\n";
        s << kTargetTag << " " << t.offset.x << " " << t.offset.y << " " << t.offset.z << " " << t.scale << " " << score.meshSize << "\n";
        return text + s.str();
    }

    bool ParseAutoStart(const std::string& text, const std::string& source, AutoStart* out, std::string* err) {
        AutoStart a;
        a.source = source;
        if (!a.design.Deserialize(text, err)) return false;
        std::istringstream in(text);
        std::string line;
        while (std::getline(in, line)) {
            std::istringstream ls(line);
            std::string tag;
            ls >> tag;
            if (tag != kTargetTag) continue;
            Placement t;
            float size = 0.0f;
            ls >> t.offset.x >> t.offset.y >> t.offset.z >> t.scale;
            if (ls.fail() || !(t.scale > 0.0f)) { *err = "Malformed " + std::string(kTargetTag) + " line: " + line.substr(0, 60); return false; }
            if (!(ls >> size)) size = 0.0f;              // (the mesh's size is optional)
            a.hasTarget = true;
            a.target = t;
            a.meshSize = size > 0.0f ? size : 0.0f;
        }
        *out = std::move(a);
        return true;
    }

    bool ReadAutoStart(const std::string& path, AutoStart* out, std::string* err) {
        std::ifstream in(path, std::ios::binary);
        if (!in) { *err = "can't open " + path; return false; }
        std::stringstream text;
        text << in.rdbuf();
        std::string name = path.substr(path.find_last_of("/\\") == std::string::npos ? 0 : path.find_last_of("/\\") + 1);
        return ParseAutoStart(text.str(), name, out, err);
    }

    // ---- the carving ----------------------------------------------------------------------------------------------------

    bool CarvingFromDesign(const Design& d, Carving* out, std::string* err, std::string* notes) {
        const std::vector<int> bodies = d.Bodies();
        if (bodies.empty()) { *err = "the design is empty"; return false; }

        struct Found {
            SuperFrustum sf;
            bool hollow = false;
            int nameId = 0;
        };
        std::vector<Found> cutters;
        int differing = 0;                               // copies of one cutter that don't match (edited in one body)
        float tol = 0.0f;                                // set from the sphere
        // the index of this cutter, found by its name's id or (without one) its shape; added when new
        auto intern = [&](const SuperFrustum& sf, bool hollow, int nameId) {
            for (size_t k = 0; k < cutters.size(); k++) {
                if (cutters[k].hollow != hollow) continue;
                bool byName = nameId > 0 && cutters[k].nameId == nameId;
                if (byName || ((nameId == 0 || cutters[k].nameId == 0) && Same(cutters[k].sf, sf, tol))) {
                    if (byName && !Same(cutters[k].sf, sf, tol)) differing++;
                    return (int)k;
                }
            }
            cutters.push_back({ sf, hollow, nameId });
            return (int)cutters.size() - 1;
        };

        struct PieceOf {
            int body = -1;                               // group id
            std::vector<int> inside, outside;            // indices into `cutters`
        };
        std::vector<PieceOf> pieces;
        std::map<int, int> pieceOfBody;                  // body group id -> index into pieces
        Vector3 centre = { 0, 0, 0 };
        float radius = 0.0f;
        int skipped = 0;                                 // shapes and features finishing added
        for (int bi : bodies) {
            const Group& body = d.groups[bi];
            // the body itself, and its "Core" if finishing wrapped it in one (a "Core 2" in another body; cores in cores
            // where it wrapped one again)
            std::vector<int> containers = { body.id };
            for (size_t k = 0; k < containers.size(); k++)
                for (int g : d.ChildGroups(containers[k]))
                    if (d.groups[g].op == Op::Union && IsNamed(d.groups[g].name, "Core")) containers.push_back(d.groups[g].id);
            const Shape* sphere = nullptr;
            for (int c : containers)
                for (int s : d.DirectShapes(c))
                    if (!sphere && IsSphere(d.shapes[s])) sphere = &d.shapes[s];
            if (!sphere) {
                *err = "body '" + body.name + "' has no generator sphere in it, so this isn't a carved sphere the generator made (only those can be "
                       "carried on from)";
                return false;
            }
            float r = sphere->sf.dilation + 0.5f * sphere->sf.size.x;
            if (pieces.empty()) {
                centre = sphere->sf.position;
                radius = r;
                tol = 1e-4f * radius;
            } else if (Vector3Distance(centre, sphere->sf.position) > 1e-3f * radius || fabsf(r - radius) > 1e-3f * radius) {
                *err = "body '" + body.name + "' is carved from a different sphere than the others";
                return false;
            }

            PieceOf piece;
            piece.body = body.id;
            for (int c : containers) {
                for (int s : d.DirectShapes(c)) {
                    const Shape& sh = d.shapes[s];
                    if (&sh == sphere) continue;
                    if (sh.op == Op::Cut && NameId(sh.name, "Cutter") > 0) piece.outside.push_back(intern(sh.sf, false, NameId(sh.name, "Cutter")));
                    else if (sh.op == Op::Cut && NameId(sh.name, "Hollow") > 0) intern(sh.sf, true, NameId(sh.name, "Hollow"));
                    else skipped++;
                }
                for (int g : d.ChildGroups(c)) {
                    const Group& f = d.groups[g];
                    if (std::find(containers.begin(), containers.end(), f.id) != containers.end()) continue;   // (a core: looked in already)
                    if (f.op != Op::Cut || !StartsWith(f.name, "Outside cutter")) { skipped++; continue; }
                    // "Around" less the cutter: the cutter is its one cut shape
                    const Shape* cut = nullptr;
                    int cuts = 0;
                    for (int s : d.DirectShapes(f.id))
                        if (d.shapes[s].op == Op::Cut) { cut = &d.shapes[s]; cuts++; }
                    if (cuts != 1 || !d.ChildGroups(f.id).empty()) {
                        *err = "'" + d.GroupPath(f.id) + "' should hold a box less one cutter, and doesn't";
                        return false;
                    }
                    int id = NameId(cut->name, "Cutter");
                    if (id == 0) id = NameId(f.name, "Outside cutter");
                    piece.inside.push_back(intern(cut->sf, false, id));
                }
            }
            pieceOfBody[body.id] = (int)pieces.size();
            pieces.push_back(std::move(piece));
        }

        // hinges: joints between pieces, which form a tree (each piece turned by one hinge at most; one root)
        struct JointOf {
            const Hinge* h;
            int parent, child;                           // indices into pieces
            int nameId;
        };
        std::vector<JointOf> joints;
        std::vector<int> turnedBy(pieces.size(), -1);
        int hardware = 0, ownOpening = 0, notClosedAtZero = 0;
        for (const Hinge& h : d.hinges) {
            auto a = pieceOfBody.find(h.parent), b = pieceOfBody.find(h.child);
            if (a == pieceOfBody.end() || b == pieceOfBody.end()) { *err = "hinge '" + h.name + "' doesn't join two of the sphere's pieces"; return false; }
            if (turnedBy[b->second] >= 0) {
                *err = "piece '" + d.groups[d.GroupIndex(h.child)].name + "' is turned by two hinges";
                return false;
            }
            turnedBy[b->second] = (int)joints.size();
            joints.push_back({ &h, a->second, b->second, NameId(h.name, "Joint") });
            hardware += h.hardware;
            ownOpening += h.ownOpening;
            notClosedAtZero += fabsf(h.end) > 1e-4f;
        }
        int roots = 0;
        for (int t : turnedBy) roots += t < 0;
        if (roots != 1) {
            *err = Utils::Format("the hinges don't join the %d pieces into one (%d of them hang from no hinge)", (int)pieces.size(), roots);
            return false;
        }
        for (size_t k = 0; k < pieces.size(); k++) {     // ... and no loops: from any piece, up its hinges to the root
            int at = (int)k, hops = 0;
            while (turnedBy[at] >= 0 && hops++ <= (int)pieces.size()) at = joints[turnedBy[at]].parent;
            if (turnedBy[at] >= 0) { *err = "the hinges go round in a loop"; return false; }
        }

        // ids: the ones the names give, where they're unique; fresh ones for the rest
        Carving c;
        c.centre = centre;
        c.radius = radius;
        std::set<int> used;
        auto claim = [&](int want) {
            if (want > 0 && used.insert(want).second) return want;
            return 0;
        };
        std::vector<int> cutterId(cutters.size()), jointId(joints.size());
        for (size_t k = 0; k < cutters.size(); k++) cutterId[k] = claim(cutters[k].nameId);
        for (size_t k = 0; k < joints.size(); k++) jointId[k] = claim(joints[k].nameId);
        int next = used.empty() ? 1 : *used.rbegin() + 1;
        for (int& id : cutterId) if (id == 0) id = next++;
        for (int& id : jointId) if (id == 0) id = next++;

        for (size_t k = 0; k < cutters.size(); k++) c.cutters.push_back({ cutterId[k], cutters[k].sf, cutters[k].hollow });
        for (const PieceOf& p : pieces) {
            Piece q;
            q.id = next++;
            for (int k : p.inside) if (std::find(q.inside.begin(), q.inside.end(), cutterId[k]) == q.inside.end()) q.inside.push_back(cutterId[k]);
            for (int k : p.outside) if (std::find(q.outside.begin(), q.outside.end(), cutterId[k]) == q.outside.end()) q.outside.push_back(cutterId[k]);
            c.pieces.push_back(q);
        }
        for (size_t k = 0; k < joints.size(); k++) {
            const Hinge& h = *joints[k].h;
            Joint j;
            j.id = jointId[k];
            j.parent = c.pieces[joints[k].parent].id;
            j.child = c.pieces[joints[k].child].id;
            j.pivot = h.pivot;
            j.axis = Vector3Length(h.axis) > 1e-6f ? Vector3Normalize(h.axis) : Vector3{ 0, 1, 0 };
            j.open = h.start;                            // (the carving's closed pose is its design pose: angle 0)
            j.t0 = h.t0;
            j.t1 = h.t1;
            c.joints.push_back(j);
        }
        // the opening order the timing shows: the first to open is the last to close (the latest t0)
        std::vector<size_t> rank(c.joints.size());
        for (size_t k = 0; k < rank.size(); k++) rank[k] = k;
        std::stable_sort(rank.begin(), rank.end(), [&](size_t a, size_t b) {
            const Joint &A = c.joints[a], &B = c.joints[b];
            return A.t0 != B.t0 ? A.t0 > B.t0 : A.t1 > B.t1;
        });
        for (size_t r = 0; r < rank.size(); r++) c.joints[rank[r]].order = (int)r;
        c.nextId = next;

        // a cutter only some piece is inside of, or only some outside of, is fine (the others were split since); one no
        // piece uses (not a hollow) would never be scored
        for (const Cutter& k : c.cutters)
            if (!k.hollow && c.PiecesUsing(k.id).empty()) { *err = Utils::Format("cutter %d shapes no piece", k.id); return false; }

        if (notes) {
            std::vector<std::string> left;
            if (hardware > 0) left.push_back(Utils::Format("hardware on %d hinge%s", hardware, hardware == 1 ? "" : "s"));
            if (!d.clearances.empty()) left.push_back(Utils::Format("%d clearance%s", (int)d.clearances.size(), d.clearances.size() == 1 ? "" : "s"));
            if (skipped > 0) left.push_back(Utils::Format("%d other shape%s and feature%s (ledges, a latch, hinge room...)", skipped, skipped == 1 ? "" : "s",
                                                          skipped == 1 ? "" : "s"));
            std::string n;
            if (!left.empty()) {
                n += "Left out what finishing added (";
                for (size_t k = 0; k < left.size(); k++) n += (k ? ", " : "") + left[k];
                n += "): the Finishing settings do it again at the end.";
            }
            if (differing > 0) n += Utils::Format("%s%d copies of a cutter differed between pieces: the first of each was used.", n.empty() ? "" : " ", differing);
            if (ownOpening > 0) n += std::string(n.empty() ? "" : " ") + "Hinges' own opening timing was dropped (the search times closing only).";
            if (notClosedAtZero > 0)
                n += Utils::Format("%s%d hinge%s didn't close at angle 0: the design pose is taken as closed.", n.empty() ? "" : " ", notClosedAtZero,
                                   notClosedAtZero == 1 ? "" : "s");
            *notes = n;
        }
        *out = std::move(c);
        return true;
    }
}
