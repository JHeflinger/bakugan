#include "progress.hpp"
#include <algorithm>

namespace Bakugan::Auto {
    void Progress::Begin(int budget, const Design& start, const AutoScore& score, int scorings) {
        m_Budget = budget;
        m_HistoryEvery = std::max(1, budget / 240);
        m_NextHistory = m_HistoryEvery;
        if (!m_Live) return;
        {
            std::lock_guard<std::mutex> l(m_Live->mutex);
            m_RecordCount = m_Live->recordFrames;
            m_Live->frames.clear();
            m_Live->iterations = budget;
            m_Live->history.clear();
            m_Live->history.push_back(score.total);
        }
        Record(start, score, scorings, false);         // the recording's first frame
    }

    void Progress::Best(const Design& best, const AutoScore& s, const Outlines& o, int scorings, bool force) {
        if (!m_Live) return;
        Clock::time_point now = Clock::now();
        if (!force && now - m_LastShown < std::chrono::milliseconds(250)) return;
        m_LastShown = now;
        Design d = best;
        m_Space.Finalize(d);
        std::vector<unsigned char> views;
        int vw = 0, vh = 0;
        if (s.silhouettes) m_Sil.Image(o, s, &views, &vw, &vh);
        std::lock_guard<std::mutex> l(m_Live->mutex);
        m_Live->views = std::move(views);
        m_Live->viewsWidth = vw;
        m_Live->viewsHeight = vh;
        m_Live->best = std::move(d);
        m_Live->score = s;
        m_Live->iteration = scorings;
        m_Live->version++;
    }

    void Progress::Tick(const Design& best, const AutoScore& score, const Counts& n) {
        m_C.SetProgress(m_Budget > 0 ? std::min(1.0f, (float)n.scorings / m_Budget) : 1.0f);
        Record(best, score, n.scorings, false);
        if (m_Live && n.scorings >= m_NextHistory) {
            m_NextHistory = n.scorings + m_HistoryEvery;
            Stats(score, n);
        }
    }

    void Progress::End(const Design& best, const AutoScore& score, const Outlines& o, const Counts& n, bool cancelled) {
        m_C.SetProgress(1.0f);
        if (!cancelled) Record(best, score, n.scorings, true);
        Best(best, score, o, n.scorings, true);
        if (m_Live) Stats(score, n);
    }

    // Snapshots of the best design at evenly spaced moments of the budget (the first now, the last at the end).
    void Progress::Record(const Design& best, const AutoScore& score, int scorings, bool last) {
        if (m_RecordCount <= 0) return;
        while (m_Recorded < m_RecordCount &&
               (last || (long long)scorings * std::max(1, m_RecordCount - 1) >= (long long)m_Recorded * std::max(1, m_Budget))) {
            AutoFrame f;
            f.design = best;
            m_Space.Finalize(f.design);
            f.score = score;
            f.iteration = std::min(scorings, m_Budget);
            std::lock_guard<std::mutex> l(m_Live->mutex);
            m_Live->frames.push_back(std::move(f));
            m_Recorded++;
        }
    }

    void Progress::Stats(const AutoScore& score, const Counts& n) {
        std::lock_guard<std::mutex> l(m_Live->mutex);
        m_Live->history.push_back(score.total);
        m_Live->iteration = n.scorings;
        m_Live->accepted = n.accepted;
        m_Live->rounds = n.rounds;
        m_Live->stepSize = n.stepSize;
    }
}
