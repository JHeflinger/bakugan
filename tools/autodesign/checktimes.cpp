// Where the collision checks go, for a few hinge timings: every hinge's turn gets the same number, however long it takes.
#include "data/clash.hpp"
#include <cstdio>
using namespace Bakugan;
static void Show(const char* what, std::vector<std::pair<float, float>> w, int n, float phase) {
    Design d;
    int prev = d.groups[d.AddGroup("Body", -1)].id;
    for (auto [a, b] : w) {
        int g = d.groups[d.AddGroup("Body", -1)].id;
        Hinge& h = d.hinges[d.AddHinge(prev, g)];
        h.t0 = a; h.t1 = b; h.start = 0; h.end = 1;
    }
    printf("%-40s phase %.2f:", what, phase);
    int last = -1;
    for (const MotionCheck& c : HingeChecks(d, n, phase)) {
        if (c.hinge != last) printf("  |");
        last = c.hinge;
        printf(" %.3f", c.t);
    }
    printf("\n");
}
int main() {
    Show("one hinge over all of it", { { 0, 1 } }, 4, 0.5f);
    Show("one hinge in 0.2..0.4", { { 0.2f, 0.4f } }, 4, 0.5f);
    Show("0.0..0.3 and 0.6..1.0", { { 0, 0.3f }, { 0.6f, 1 } }, 4, 0.5f);
    Show("0..1 and 0.40..0.45 (a quick one)", { { 0, 1 }, { 0.4f, 0.45f } }, 4, 0.5f);
    Show("0..1 and 0.40..0.45 (a quick one)", { { 0, 1 }, { 0.4f, 0.45f } }, 4, 0.0f);
}
