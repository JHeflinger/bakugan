// Where a search starts: the mesh split into hinged pieces, or the design in the editor.
#include "search.hpp"
#include <algorithm>
#include <cfloat>

namespace Bakugan::Auto {
    // Splits the inside of the mesh into k pieces (k-means), gives each a box-ish superfrustum along its principal
    // axes, and hinges them in a tree (nearest neighbours first, from the biggest piece) with each piece folding
    // towards the middle of the mesh.
    Design Search::Seed() {
        Design d;
        int k = std::clamp(std::max(m_P.seedBodies, m_Space.minBodies), 1, m_Space.maxBodies);
        std::vector<Vector3> pts = m_T.inside;
        std::shuffle(pts.begin(), pts.end(), m_Rng);
        if (pts.size() > 2500) pts.resize(2500);
        k = std::max(std::min(k, (int)pts.size() / 20), 1);

        // k-means++: centres drawn far from the ones already drawn, then 15 rounds of k-means
        std::vector<Vector3> centres = { PickFrom(m_Rng, pts) };
        std::vector<float> dist(pts.size(), FLT_MAX);
        while ((int)centres.size() < k) {
            double total = 0.0;
            for (size_t i = 0; i < pts.size(); i++) {
                dist[i] = fminf(dist[i], Vector3DistanceSqr(pts[i], centres.back()));
                total += dist[i];
            }
            double r = Uniform(m_Rng) * total;
            size_t pick = 0;
            for (; pick + 1 < pts.size() && r > dist[pick]; pick++) r -= dist[pick];
            centres.push_back(pts[pick]);
        }
        std::vector<int> label(pts.size(), 0);
        for (int it = 0; it < 15; it++) {
            for (size_t i = 0; i < pts.size(); i++) {
                float best = FLT_MAX;
                for (int c = 0; c < k; c++) {
                    float dd = Vector3DistanceSqr(pts[i], centres[c]);
                    if (dd < best) { best = dd; label[i] = c; }
                }
            }
            std::vector<Vector3> sum(k, { 0, 0, 0 });
            std::vector<int> count(k, 0);
            for (size_t i = 0; i < pts.size(); i++) { sum[label[i]] = Vector3Add(sum[label[i]], pts[i]); count[label[i]]++; }
            for (int c = 0; c < k; c++)
                if (count[c] > 0) centres[c] = Vector3Scale(sum[c], 1.0f / count[c]);
        }
        std::vector<std::vector<Vector3>> cluster(k);
        for (size_t i = 0; i < pts.size(); i++) cluster[label[i]].push_back(pts[i]);
        std::vector<int> order;                 // the pieces worth a body, biggest first
        for (int c = 0; c < k; c++)
            if (cluster[c].size() >= 8) order.push_back(c);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return cluster[a].size() > cluster[b].size(); });

        // one oriented, trimmed box per piece
        std::vector<int> bodyOf(k, -1);
        for (int c : order) {
            SuperFrustum f = BoxAround(cluster[c], 0.03f, 0.02f * m_T.L);
            f.rounding = 0.6f;
            int b = d.groups[d.AddGroup("Body", -1)].id;
            d.AddShape(f, b, "Shape");
            bodyOf[c] = b;
        }

        // the hinge tree: Prim's algorithm over the gaps between pieces, from the biggest
        auto gapBetween = [&](int a, int b, Vector3* mid) {
            float best = FLT_MAX;
            size_t na = std::min(cluster[a].size(), (size_t)160), nb = std::min(cluster[b].size(), (size_t)160);
            for (size_t i = 0; i < na; i++)
                for (size_t j = 0; j < nb; j++) {
                    float dd = Vector3DistanceSqr(cluster[a][i], cluster[b][j]);
                    if (dd < best) { best = dd; *mid = Vector3Lerp(cluster[a][i], cluster[b][j], 0.5f); }
                }
            return best;
        };
        std::vector<int> tree;
        if (!order.empty()) tree.push_back(order[0]);
        while (tree.size() < order.size()) {
            float best = FLT_MAX;
            int bp = -1, bc = -1;
            Vector3 bmid = { 0, 0, 0 };
            for (int a : tree)
                for (int c : order) {
                    if (std::find(tree.begin(), tree.end(), c) != tree.end()) continue;
                    Vector3 mid = { 0, 0, 0 };
                    float g = gapBetween(a, c, &mid);
                    if (g < best) { best = g; bp = a; bc = c; bmid = mid; }
                }
            if (bc < 0) break;
            tree.push_back(bc);
            int hi = d.HingeIndex(m_Grammar.AttachHinge(d, bodyOf[bp], bodyOf[bc], bmid, centres[bc]));
            if (hi < 0) continue;
            // seeds always fold towards the middle, most of the way
            Hinge& hg = d.hinges[hi];
            Vector3 out = Vector3Normalize(Vector3Subtract(centres[bc], bmid));
            Vector3 in = Vector3Subtract(m_T.centre, bmid);
            Vector3 axis = Vector3CrossProduct(out, in);
            if (Vector3Length(axis) > 1e-3f * fmaxf(Vector3Length(in), 1e-3f)) {
                hg.axis = Vector3Normalize(axis);
                hg.end = 0.85f * acosf(std::clamp(Vector3DotProduct(out, Vector3Normalize(in)), -1.0f, 1.0f));
            }
        }
        return d;
    }

    Design Search::FromEditor(const Design& start) {
        Design d = start;
        d.Repair();
        m_Space.Allow(d);
        return d;
    }
}
