#pragma once
// A design's bodies as signed distance fields, ready to be sampled by the score, and what a score reports back to
// the grammar about where the design is wrong.
#include "common.hpp"
#include "data/field.hpp"

namespace Bakugan::Auto {
    // The bodies of a design as fields. Distances are capped: anything farther than the cap counts the same, which
    // lets whole bodies be skipped by their boxes.
    struct Probe {
        std::vector<BodyField> fields;
        std::vector<int> group;              // group index of each body
        std::vector<int> id;                 // group id
        std::vector<BoundingBox> box;        // design pose

        void Build(const Design& d, float margin) {
            for (int g : d.Bodies()) {
                BodyField f = BodyField::Body(d, d.groups[g].id);
                if (f.Empty()) continue;
                BoundingBox b = d.BodyBounds(d.groups[g].id);
                float blend = 0.0f;
                for (int c : d.ChildGroups(d.groups[g].id)) blend = fmaxf(blend, d.groups[c].blend);
                box.push_back(Grow(b, fmaxf(blend, d.groups[g].blend) + margin));
                fields.push_back(std::move(f));
                group.push_back(g);
                id.push_back(d.groups[g].id);
            }
        }

        int Index(int groupId) const {
            for (int i = 0; i < (int)id.size(); i++)
                if (id[i] == groupId) return i;
            return -1;
        }

        // Distance to the union of the bodies, each moved by xf (indexed like the groups; null: design pose), and
        // the body nearest (group index).
        float Distance(Vector3 p, const std::vector<Rigid>* xf, float cap, int* who = nullptr) const {
            float best = cap;
            int w = -1;
            for (int i = 0; i < (int)fields.size(); i++) {
                Vector3 q = xf ? (*xf)[group[i]].ApplyInverse(p) : p;
                float bd = BoxDistance(q, box[i]);
                if (bd > 0.0f && bd >= best) continue;
                float d = fields[i].Distance(q);
                if (d < best) { best = d; w = i; }
            }
            if (who) *who = w >= 0 ? group[w] : -1;
            return best;
        }

        // The nearest two bodies: d1 is the union's distance, d2 the next body's (negative where two bodies overlap).
        void Distance2(Vector3 p, const std::vector<Rigid>* xf, float cap, float* d1, float* d2, int* w1, int* w2) const {
            float b1 = cap, b2 = cap;
            int i1 = -1, i2 = -1;
            for (int i = 0; i < (int)fields.size(); i++) {
                Vector3 q = xf ? (*xf)[group[i]].ApplyInverse(p) : p;
                float bd = BoxDistance(q, box[i]);
                if (bd > 0.0f && bd >= b2) continue;
                float d = fields[i].Distance(q);
                if (d < b1) { b2 = b1; i2 = i1; b1 = d; i1 = i; }
                else if (d < b2) { b2 = d; i2 = i; }
            }
            *d1 = b1;
            *d2 = b2;
            *w1 = i1 >= 0 ? group[i1] : -1;
            *w2 = i2 >= 0 ? group[i2] : -1;
        }
    };

    // A spot the last score found wrong (world position, in the pose it was found in) and the body (group index)
    // nearest to it there.
    struct Spot {
        Vector3 p = { 0, 0, 0 };
        int body = -1;
    };

    // Two bodies overlapping at one moment of the motion (0 open, 1 closed; of the opening motion or the closing one).
    struct Clash {
        Vector3 p = { 0, 0, 0 };
        int a = -1, b = -1;              // group indices
        float t = 0.0f;
        bool opening = false;
    };

    // Where a design is wrong, found while scoring it: the grammar proposes its data-driven rewrites there.
    struct Diagnostics {
        std::vector<Spot> missing;       // open pose: inside the mesh, outside the design
        std::vector<Spot> excess;        // open pose: inside the design, outside the mesh
        std::vector<Spot> holes;         // closed pose: inside the sphere, outside the closed shape
        std::vector<Spot> bumps;         // closed pose: inside the closed shape, outside the sphere
        std::vector<Clash> clashes;      // bodies overlapping, in the end poses or on the way
    };
}
