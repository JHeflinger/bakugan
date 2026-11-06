#include "autodesign/suite.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
using namespace Bakugan;
int main(int argc, char** argv) {
    SuiteParams p;
    p.folder = argv[1];
    p.generate.steps = argc > 2 ? atoi(argv[2]) : 1500;
    p.videoSize = argc > 3 ? atoi(argv[3]) : 720;
    p.searchSeconds = argc > 4 ? atof(argv[4]) : 4.0f;
    JobControl c;
    auto live = std::make_shared<SuiteLive>();
    auto t0 = std::chrono::steady_clock::now();
    SuiteResult r = RunSuite(p, c, live);
    printf("%s\n%.1f s\n", r.message.c_str(), std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return r.ok ? 0 : 1;
}
