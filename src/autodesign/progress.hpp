#pragma once
// How a running search is shown: the shared AutoLive the UI reads (best design, its outlines, the score history),
// the job's progress bar, and evenly spaced snapshots for a recording of the run.
#include "score.hpp"
#include "space.hpp"
#include <memory>

namespace Bakugan::Auto {
    class Progress {
    public:
        Progress(std::shared_ptr<AutoLive> live, JobControl& c, const DesignSpace& space, const Silhouettes& sil)
            : m_Live(std::move(live)), m_C(c), m_Space(space), m_Sil(sil) {}

        // The search's counters, as shown.
        struct Counts {
            int scorings = 0, accepted = 0, rounds = 0;
            float stepSize = 0.0f;
        };

        void Begin(int budget, const Design& start, const AutoScore& score, int scorings);
        // A new best design: shown at most every quarter second (force: now).
        void Best(const Design& d, const AutoScore& s, const Outlines& o, int scorings, bool force);
        // After every step: the progress bar, the history plot, the recording.
        void Tick(const Design& best, const AutoScore& score, const Counts& n);
        void End(const Design& best, const AutoScore& score, const Outlines& o, const Counts& n, bool cancelled);
    private:
        void Record(const Design& best, const AutoScore& score, int scorings, bool last);
        void Stats(const AutoScore& score, const Counts& n);
    private:
        std::shared_ptr<AutoLive> m_Live;
        JobControl& m_C;
        const DesignSpace& m_Space;
        const Silhouettes& m_Sil;
        int m_Budget = 0;
        int m_RecordCount = 0, m_Recorded = 0;
        int m_HistoryEvery = 1, m_NextHistory = 1;
        Clock::time_point m_LastShown;
    };
}
