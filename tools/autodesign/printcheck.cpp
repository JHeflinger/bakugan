// Whether saved designs can be printed and stay in one piece: ./printcheck design.txt ...
// (every body one solid, every body but the root on a hinge whose hardware is part of both bodies: CheckPrintable)
#include "data/finish.hpp"
#include <cstdio>
#include <fstream>
#include <sstream>
using namespace Bakugan;

int main(int argc, char** argv) {
    int bad = 0;
    for (int k = 1; k < argc; k++) {
        std::ifstream in(argv[k]);
        std::stringstream text;
        text << in.rdbuf();
        Design d;
        std::string err;
        if (!in || !d.Deserialize(text.str(), &err)) { printf("%s: can't read it: %s\n", argv[k], err.c_str()); bad++; continue; }
        PrintCheck c = CheckPrintable(d);
        printf("%s: %s\n", argv[k], c.message.c_str());
        bad += !c.ok;
    }
    return bad > 0;
}
