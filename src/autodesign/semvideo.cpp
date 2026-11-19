#include "semvideo.hpp"
#include "loss.hpp"
#include "util/utils.hpp"
#include <algorithm>

namespace Bakugan {
    using namespace Suite;

    void SemanticVideo::Add(const Loss& loss, const std::string& caption, const AutoScore& s, float progress, int hold) {
        if (m_Failed) return;
        std::vector<unsigned char> rgba;
        int pw = 0, ph = 0;
        loss.SemanticPicture(&rgba, &pw, &ph);
        if (pw <= 0 || ph <= 0) return;
        // the picture enlarged (whole pixels) to about 1280 across, a label column, a caption and a progress bar above
        const int k = std::clamp(1280 / pw, 1, 6), ts = 2, pad = 12, line = 10 * ts + 6;
        const int labelW = std::max(TextWidth("Mesh", ts), TextWidth("Design", ts)) + 2 * pad;
        int w = labelW + pw * k + pad, h = pad + 2 * line + 8 + pad + ph * k + pad;
        w += w & 1;
        h += h & 1;
        if (m_Frames == 0) {
            m_W = w;
            m_H = h;
            if (!m_Encoder.Open(m_Path, m_W, m_H, m_Fps)) {
                m_Failed = true;
                m_Error = "could not start ffmpeg";
                return;
            }
        }
        if (w != m_W || h != m_H) return;                // (the views can't change size in a run)
        Canvas cv;
        cv.Resize(m_W, m_H);
        int y = pad;
        cv.Text(pad, y, ts, caption, Ink);
        std::string scores = Utils::Format("loss %.4f   semantic %.4f (worst view %.4f)   fit %.4f", s.total, s.semantic, s.worstSemantic, s.fit);
        cv.Text(pad, y + line, ts, FitText(scores, m_W - 2 * pad, ts), Muted);
        y += 2 * line;
        cv.Fill(pad, y, m_W - pad, y + 6, Track);
        cv.Fill(pad, y, pad + (int)((m_W - 2 * pad) * std::clamp(progress, 0.0f, 1.0f)), y + 6, Accent);
        y += 8 + pad;
        // the two rows of the picture (SemanticPicture: a 3 pixel gap, the mesh's row, a gap, the design's row, a gap)
        const int rowH = (ph - 9) / 2;
        cv.Text(pad, y + (3 + rowH / 2) * k - 5 * ts, ts, "Mesh", Muted);
        cv.Text(pad, y + (6 + rowH + rowH / 2) * k - 5 * ts, ts, "Design", Muted);
        for (int py = 0; py < ph * k && y + py < m_H; py++)
            for (int px = 0; px < pw * k && labelW + px < m_W; px++) {
                const unsigned char* q = &rgba[((size_t)(py / k) * pw + px / k) * 4];
                cv.px[(size_t)(y + py) * m_W + labelW + px] = { q[0], q[1], q[2] };
            }
        cv.Rgb24(&m_Rgb);
        for (int r = 0; r < std::max(1, hold); r++) {
            if (!m_Encoder.Write(m_Rgb)) {
                m_Failed = true;
                m_Error = "ffmpeg stopped taking frames";
                return;
            }
            m_Frames++;
        }
    }

    bool SemanticVideo::Finish(std::string* message) {
        bool ok = m_Frames > 0 && !m_Failed;
        if (m_Encoder.IsOpen()) ok = m_Encoder.Close() && ok;
        if (m_Frames == 0 && m_Error.empty()) m_Error = "no frames: the semantic loss drew no pictures";
        if (ok) *message = Utils::Format("Semantic video: %d frames (%.1f s) in %s.", m_Frames, (float)m_Frames / m_Fps, m_Path.c_str());
        else *message = "No semantic video: " + (m_Error.empty() ? std::string("ffmpeg failed (is libx264 available?)") : m_Error) + ".";
        return ok;
    }
}
