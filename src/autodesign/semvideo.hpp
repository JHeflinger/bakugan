#pragma once
// A video of the semantic comparison over a run: every so many steps, the views DINOv3 compared (the placed mesh's
// above, the design's below, as in the Generate tab) with the step and the scores, streamed into ffmpeg as they come.
#include "autodesign.hpp"
#include "suite_draw.hpp"
#include <string>

namespace Bakugan {
    class Loss;

    class SemanticVideo {
    public:
        SemanticVideo(const std::string& path, int fps) : m_Path(path), m_Fps(std::max(1, fps)) {}
        // One frame of the loss's latest semantic pictures (shown `hold` frames). The first frame sets the size and
        // starts ffmpeg. Does nothing once something failed.
        void Add(const Loss& loss, const std::string& caption, const AutoScore& s, float progress, int hold = 1);
        // ffmpeg finished: true with the video written; a sentence saying how it went either way.
        bool Finish(std::string* message);
        int Fps() const { return m_Fps; }
    private:
        std::string m_Path, m_Error;
        int m_Fps;
        int m_Frames = 0, m_W = 0, m_H = 0;
        bool m_Failed = false;
        Suite::Encoder m_Encoder;
        std::vector<unsigned char> m_Rgb;
    };
}
