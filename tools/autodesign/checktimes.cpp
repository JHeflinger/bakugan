#include "autodesign/loss.hpp"
#include <cstdio>
using namespace Bakugan;
static void Show(const char* what, std::vector<std::pair<float, float>> w, int n, float bias) {
    Design d;
    int prev = d.groups[d.AddGroup("Body", -1)].id;
    for (auto [a, b] : w) {
        int g = d.groups[d.AddGroup("Body", -1)].id;
        Hinge& h = d.hinges[d.AddHinge(prev, g)];
        h.t0 = a; h.t1 = b; h.start = 0; h.end = 1;
    }
    printf("%-40s bias %.1f:", what, bias);
    for (float t : CheckTimes(d, n, bias)) printf(" %.3f", t);
    printf("\n");
}
int main() {
    Show("one hinge over all of it", { { 0, 1 } }, 8, 0.5f);
    Show("one hinge in 0.2..0.4", { { 0.2f, 0.4f } }, 8, 0.5f);
    Show("0.0..0.3 and 0.6..1.0 (gap between)", { { 0, 0.3f }, { 0.6f, 1 } }, 8, 0.5f);
    Show("0..1 and 0.4..0.6 (two at once mid)", { { 0, 1 }, { 0.4f, 0.6f } }, 10, 0.0f);
    Show("0..1 and 0.4..0.6 (two at once mid)", { { 0, 1 }, { 0.4f, 0.6f } }, 10, 0.5f);
    Show("0..1 and 0.4..0.6 (two at once mid)", { { 0, 1 }, { 0.4f, 0.6f } }, 10, 1.0f);
}
