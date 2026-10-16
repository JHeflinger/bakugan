#include "design.hpp"
#include "util/utils.hpp"
#include <raymath.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>

namespace Bakugan {
    static bool SameVec(Vector3 a, Vector3 b) { return a.x == b.x && a.y == b.y && a.z == b.z; }

    bool Hinge::operator==(const Hinge& o) const {
        return parent == o.parent && SameVec(pivot, o.pivot) && SameVec(axis, o.axis) && angle == o.angle &&
               maxAngle == o.maxAngle && locked == o.locked;
    }

    bool Part::operator==(const Part& o) const {
        return id == o.id && name == o.name && shape == o.shape && hinge == o.hinge && color.r == o.color.r &&
               color.g == o.color.g && color.b == o.color.b && color.a == o.color.a;
    }

    bool Ball::operator==(const Ball& o) const {
        return SameVec(center, o.center) && radius == o.radius;
    }

    bool Design::operator==(const Design& o) const {
        return parts == o.parts && ball == o.ball && nextId == o.nextId;
    }

    int Design::Root() const {
        for (int i = 0; i < (int)parts.size(); i++)
            if (parts[i].hinge.parent < 0) return i;
        return -1;
    }

    int Design::IndexOf(int id) const {
        for (int i = 0; i < (int)parts.size(); i++)
            if (parts[i].id == id) return i;
        return -1;
    }

    float Design::FoldAngle(int i) const {
        const Hinge& h = parts[i].hinge;
        if (h.parent < 0 || h.locked) return 0.0f;
        return h.angle;
    }

    std::vector<Rigid> Design::Transforms(float t) const {
        int n = (int)parts.size();
        std::vector<Rigid> xf(n);
        std::vector<char> done(n, 0);
        for (int i : TopologicalOrder()) {
            const Hinge& h = parts[i].hinge;
            Rigid local = Rigid::AboutAxis(h.pivot, h.axis, FoldAngle(i) * t);
            xf[i] = (h.parent >= 0 && done[h.parent]) ? local.Then(xf[h.parent]) : local;
            done[i] = 1;
        }
        return xf;
    }

    void Design::HingeInWorld(int i, const std::vector<Rigid>& xf, Vector3* pivot, Vector3* axis) const {
        const Hinge& h = parts[i].hinge;
        if (h.parent < 0) { *pivot = h.pivot; *axis = h.axis; return; }
        *pivot = xf[h.parent].Apply(h.pivot);
        *axis = xf[h.parent].ApplyVector(h.axis);
    }

    std::vector<int> Design::Children(int part) const {
        std::vector<int> out;
        for (int i = 0; i < (int)parts.size(); i++)
            if (parts[i].hinge.parent == part) out.push_back(i);
        return out;
    }

    bool Design::IsDescendant(int part, int ancestor) const {
        int guard = 0;
        for (int p = part; p >= 0 && guard <= (int)parts.size(); p = parts[p].hinge.parent, guard++)
            if (p == ancestor) return true;
        return false;
    }

    std::vector<int> Design::Subtree(int part) const {
        std::vector<int> out;
        for (int i = 0; i < (int)parts.size(); i++)
            if (IsDescendant(i, part)) out.push_back(i);
        return out;
    }

    int Design::Depth(int part) const {
        int d = 0;
        for (int p = parts[part].hinge.parent; p >= 0 && d <= (int)parts.size(); p = parts[p].hinge.parent) d++;
        return d;
    }

    std::vector<int> Design::TopologicalOrder() const {
        int n = (int)parts.size();
        std::vector<int> order, depth(n);
        for (int i = 0; i < n; i++) depth[i] = Depth(i);
        order.resize(n);
        for (int i = 0; i < n; i++) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return depth[a] < depth[b]; });
        return order;
    }

    int Design::AddPart(const SuperFrustum& shape, int parent, const std::string& name) {
        Part p;
        p.id = nextId++;
        p.name = name.empty() ? "part_" + std::to_string(p.id) : name;
        p.shape = shape;
        p.color = Utils::PartColor(p.id);
        p.hinge.parent = parent;
        p.hinge.pivot = shape.position;
        parts.push_back(p);
        return (int)parts.size() - 1;
    }

    void Design::RemovePart(int part) {
        if (part < 0 || part >= (int)parts.size()) return;
        int up = parts[part].hinge.parent;
        for (Part& p : parts)
            if (p.hinge.parent == part) p.hinge.parent = up;
        parts.erase(parts.begin() + part);
        for (Part& p : parts)
            if (p.hinge.parent > part) p.hinge.parent--;
    }

    bool Design::Reparent(int part, int parent) {
        if (part < 0 || part >= (int)parts.size() || parent >= (int)parts.size()) return false;
        if (parent >= 0 && IsDescendant(parent, part)) return false;   // would make a cycle
        parts[part].hinge.parent = parent;
        return true;
    }

    void Design::Reroot(int part) {
        if (part < 0 || part >= (int)parts.size()) return;
        // Folding c about (pivot, axis, a) relative to p is the same as folding p about (pivot, -axis, a) relative to c,
        // so each hinge on the path flips over to the other part. In the open pose all frames coincide.
        // the new root stays put while folding, so the sphere moves into its frame
        ball.center = Transforms(1.0f)[part].ApplyInverse(ball.center);
        std::vector<int> path;
        int guard = 0;
        for (int p = part; p >= 0 && guard++ <= (int)parts.size(); p = parts[p].hinge.parent) path.push_back(p);
        std::vector<Hinge> old(path.size());
        for (size_t k = 0; k < path.size(); k++) old[k] = parts[path[k]].hinge;
        parts[part].hinge.parent = -1;
        for (size_t k = 0; k + 1 < path.size(); k++) {
            Hinge h = old[k];
            h.parent = path[k];
            h.axis = Vector3Negate(h.axis);
            parts[path[k + 1]].hinge = h;
        }
        // keep the root first in the list, since the first parentless part is the root
        if (part != 0) {
            std::swap(parts[0], parts[part]);
            for (Part& p : parts) {
                if (p.hinge.parent == 0) p.hinge.parent = part;
                else if (p.hinge.parent == part) p.hinge.parent = 0;
            }
        }
    }

    int Design::CountHinges() const {
        int n = 0;
        for (const Part& p : parts)
            if (p.hinge.parent >= 0 && !p.hinge.locked) n++;
        return n;
    }

    void Design::RepairHinges() {
        for (Part& p : parts) {
            Hinge& h = p.hinge;
            float len = Vector3Length(h.axis);
            h.axis = len > 1e-6f ? Vector3Scale(h.axis, 1.0f / len) : Vector3{ 1, 0, 0 };
            h.maxAngle = std::clamp(h.maxAngle, 0.0f, PI);
            h.angle = std::clamp(h.angle, 0.0f, h.maxAngle);
            if (h.locked) h.angle = 0.0f;
        }
        // break cycles defensively
        for (int i = 0; i < (int)parts.size(); i++) {
            int guard = 0;
            for (int p = parts[i].hinge.parent; p >= 0; p = parts[p].hinge.parent)
                if (++guard > (int)parts.size()) { parts[i].hinge.parent = -1; break; }
        }
    }

    std::string Design::Serialize() const {
        std::ostringstream s;
        s.precision(9);
        s << "bakugan-design 1\n";
        s << "ball " << ball.center.x << " " << ball.center.y << " " << ball.center.z << " " << ball.radius << "\n";
        s << "nextid " << nextId << "\n";
        for (const Part& p : parts) {
            const SuperFrustum& f = p.shape;
            const Hinge& h = p.hinge;
            std::string name = p.name;
            std::replace(name.begin(), name.end(), ' ', '_');
            s << "part " << p.id << " " << name << " " << (int)p.color.r << " " << (int)p.color.g << " " << (int)p.color.b
              << " sf " << f.position.x << " " << f.position.y << " " << f.position.z << " "
              << f.rotation.x << " " << f.rotation.y << " " << f.rotation.z << " " << f.rotation.w << " "
              << f.size.x << " " << f.size.y << " " << f.size.z << " "
              << f.rounding << " " << f.taper << " " << f.bulge << " " << f.dilation << " " << f.onion
              << " hinge " << h.parent << " " << h.pivot.x << " " << h.pivot.y << " " << h.pivot.z << " "
              << h.axis.x << " " << h.axis.y << " " << h.axis.z << " " << h.angle << " " << h.maxAngle << " "
              << (h.locked ? 1 : 0) << "\n";
        }
        return s.str();
    }

    bool Design::Deserialize(const std::string& text, std::string* err) {
        std::istringstream in(text);
        std::string line;
        Design d;
        d.parts.clear();
        bool header = false;
        while (std::getline(in, line)) {
            std::istringstream ls(line);
            std::string tag;
            ls >> tag;
            if (tag == "bakugan-design") header = true;
            else if (tag == "ball") ls >> d.ball.center.x >> d.ball.center.y >> d.ball.center.z >> d.ball.radius;
            else if (tag == "nextid") ls >> d.nextId;
            else if (tag == "part") {
                Part p;
                int r, g, b;
                std::string sf, hg;
                SuperFrustum& f = p.shape;
                Hinge& h = p.hinge;
                int locked = 0;
                ls >> p.id >> p.name >> r >> g >> b >> sf
                   >> f.position.x >> f.position.y >> f.position.z
                   >> f.rotation.x >> f.rotation.y >> f.rotation.z >> f.rotation.w
                   >> f.size.x >> f.size.y >> f.size.z
                   >> f.rounding >> f.taper >> f.bulge >> f.dilation >> f.onion
                   >> hg >> h.parent >> h.pivot.x >> h.pivot.y >> h.pivot.z
                   >> h.axis.x >> h.axis.y >> h.axis.z >> h.angle >> h.maxAngle >> locked;
                if (ls.fail() || sf != "sf" || hg != "hinge") { *err = "Malformed part line: " + line.substr(0, 60); return false; }
                p.color = { (unsigned char)r, (unsigned char)g, (unsigned char)b, 255 };
                h.locked = locked != 0;
                d.parts.push_back(p);
            }
        }
        if (!header) { *err = "Not a bakugan design file"; return false; }
        for (Part& p : d.parts)
            if (p.hinge.parent >= (int)d.parts.size()) p.hinge.parent = -1;
        for (const Part& p : d.parts) d.nextId = std::max(d.nextId, p.id + 1);
        d.RepairHinges();
        *this = d;
        return true;
    }
}
