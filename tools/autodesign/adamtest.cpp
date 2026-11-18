// Adam keeps a parameter's memory while it exists, gives new ones fresh memory, drops gone ones.
#include "autodesign/adam.hpp"
#include <cstdio>
using namespace Bakugan;
int main() {
    Adam a; a.lr = 0.1f;
    std::vector<uint64_t> k = { 1, 2 };
    std::vector<float> x = { 0, 0 };
    for (int i = 0; i < 5; i++) a.Step(k, x, { 1.0f, -1.0f });
    printf("after 5 steps: x = %.3f %.3f, %zu slots\n", x[0], x[1], a.Slots());
    k = { 1, 2, 3 }; x.push_back(0);                       // a new parameter
    a.Step(k, x, { 1.0f, -1.0f, 1.0f });
    printf("added 3: x = %.3f %.3f %.3f, %zu slots (its first step is a full lr, like the others' first)\n", x[0], x[1], x[2], a.Slots());
    k = { 1, 3 }; x = { x[0], x[2] };                       // parameter 2 gone
    a.Step(k, x, { 0.0f, 0.0f });
    printf("removed 2: %zu slots; with zero gradient 1 still moves on momentum: x = %.3f %.3f\n", a.Slots(), x[0], x[1]);
}
