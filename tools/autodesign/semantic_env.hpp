#pragma once
// The semantic loss's settings from the environment, for the test tools:
//   SEM=model.onnx  turns it on (weight 1)        SEMW=weight      ORT=path/to/libonnxruntime
//   LOOK=0|1|2      silhouette, depth, shaded     PATCH=0..1       0 global embeddings .. 1 patch embeddings
//   SILW=weight     the silhouettes' weight       SAMPLES=n        shaded pixels sampled for the gradient
//   GPU=1           CUDA, if the runtime has it
#include "autodesign/autodesign.hpp"
#include <cstdlib>

inline void SemanticFromEnv(Bakugan::AutoParams& p) {
    if (const char* m = getenv("SEM")) { p.semanticModel = m; p.semanticWeight = 1.0f; }
    if (const char* s = getenv("SEMW")) p.semanticWeight = (float)atof(s);
    if (const char* s = getenv("ORT")) p.onnxRuntime = s;
    if (const char* s = getenv("LOOK")) p.semanticLook = (Bakugan::SemanticLook)atoi(s);
    if (const char* s = getenv("PATCH")) p.patchWeight = (float)atof(s);
    if (const char* s = getenv("SILW")) p.silhouetteWeight = (float)atof(s);
    if (const char* s = getenv("SAMPLES")) p.semanticSamples = atoi(s);
    if (getenv("GPU")) p.semanticGpu = true;
}
