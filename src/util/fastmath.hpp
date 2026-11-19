#pragma once
// min and max of floats that compile to one instruction. fminf/fmaxf are library calls under the default flags
// (their NaN rules don't match the instructions), and the distance fields make billions of them. No NaNs reach these.
namespace Bakugan {
    inline float Fmin(float a, float b) { return b < a ? b : a; }
    inline float Fmax(float a, float b) { return b > a ? b : a; }
}
