#pragma once
#include "data/job.hpp"
#include "data/shape.hpp"
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace Bakugan {
    struct FoldParams {
        float volumeTolerance = 0.15f;   // the ball radius may stray this far (relative) from the equal-volume radius
        float maxHingeAngle = PI;        // largest fold a hinge may make
        int iterations = 220;            // continuous descent steps
        int rewriteEvery = 6;            // a grammar rewrite phase after this many steps
        int candidates = 24;             // K: rewrites proposed per phase
        bool restructure = true;         // use the grammar at all
        bool keepTree = false;           // no reparent / split / merge (keeps a hand-made tree)
        bool allowSplit = true;
        bool checkPath = true;           // also penalise collisions half way through the fold
        float wCoverage = 2.5f;          // sphere shell left uncovered
        float wShellOverlap = 0.6f;      // shell covered twice
        float wTrim = 1.0f;              // material cut off by the sphere
        float wCollision = 2.5f;         // material of different parts in the same place when closed
        float wPath = 0.3f;              // ... half way through folding
        float wDetail = 1.0f;            // extra weight of surface detail (curvature) in the trim term
        float wContact = 0.5f;           // hinge pivots must sit inside both parts
        float wVolume = 0.4f;            // keep the sphere near the equal-volume size (within the tolerance band)
        float wLink = 1.0f;              // hinge pivots must stay inside the closed sphere, or the chain breaks
        float hingeCost = 0.004f;        // simplicity: cost per working hinge
        int samplesPerPart = 260;
        int shellSamples = 640;
        int maxParts = 24;
        unsigned seed = 7;

        bool operator==(const FoldParams& o) const {
            return volumeTolerance == o.volumeTolerance && maxHingeAngle == o.maxHingeAngle && iterations == o.iterations &&
                   rewriteEvery == o.rewriteEvery && candidates == o.candidates && restructure == o.restructure &&
                   keepTree == o.keepTree && allowSplit == o.allowSplit && checkPath == o.checkPath &&
                   wCoverage == o.wCoverage && wShellOverlap == o.wShellOverlap && wTrim == o.wTrim &&
                   wCollision == o.wCollision && wPath == o.wPath && wDetail == o.wDetail && wContact == o.wContact &&
                   wVolume == o.wVolume && wLink == o.wLink && hingeCost == o.hingeCost && samplesPerPart == o.samplesPerPart &&
                   shellSamples == o.shellSamples && maxParts == o.maxParts && seed == o.seed;
        }
        bool operator!=(const FoldParams& o) const { return !(*this == o); }
    };

    struct FoldMetrics {
        bool valid = false;
        float coverage = 0;              // fraction of the sphere shell covered by a part
        float shellOverlap = 0;          // fraction of the shell claimed by two or more parts
        float trimmed = 0;               // fraction of (weighted) material outside the sphere when closed
        float collision = 0;             // fraction of material samples that collide when closed
        float pathCollision = 0;
        float contact = 0;               // hinge pivots outside their parts (relative)
        float loss = 0;
        int hinges = 0;
        float radius = 0;
        float radiusRatio = 1;           // radius / equal-volume radius
    };

    // Positions for the viewport overlays (closed pose).
    struct FoldVisual {
        std::vector<Vector3> shell;
        std::vector<uint8_t> shellState; // 0 uncovered, 1 covered, 2 covered twice
        std::vector<Vector3> lost;       // material samples trimmed or colliding when closed
    };

    // Material of the input shape attached to one part (open pose).
    struct PartSamples {
        std::vector<Vector3> lattice;    // interior points on a regular lattice
        std::vector<DetailSample> detail;
    };

    struct FoldState {
        Design design;
        std::vector<PartSamples> samples;   // parallel to design.parts
    };

    // The objective of the folding problem. The material of the input is sampled once; each part owns the samples
    // closest to its superfrustum. Superfrustums act as the parts' bounding volumes for the sphere shell.
    class FoldModel {
    public:
        FoldModel() = default;
        FoldModel(std::shared_ptr<const ShapeAnalysis> shape, const FoldParams& params);
        bool Valid() const { return m_Shape != nullptr; }
        const FoldParams& Params() const { return m_Params; }
        void SetParams(const FoldParams& p);
        float Spacing() const { return m_Spacing; }
        float EqualRadius() const { return m_Shape ? m_Shape->equalRadius : 1.0f; }
        Vector3 Centroid() const { return m_Shape ? m_Shape->centroid : Vector3{ 0, 1, 0 }; }

        FoldState MakeState(const Design& d) const;
        void Assign(FoldState& s) const;
        float Evaluate(const Design& d, const std::vector<PartSamples>& samples, FoldMetrics* metrics = nullptr,
                       FoldVisual* visual = nullptr) const;
        float Evaluate(const FoldState& s, FoldMetrics* metrics = nullptr, FoldVisual* visual = nullptr) const {
            return Evaluate(s.design, s.samples, metrics, visual);
        }
        float CollisionLoss(const std::vector<PartSamples>& samples, const std::vector<Rigid>& xf, float* fraction,
                            std::vector<Vector3>* lost, int stride = 1) const;
        float RadiusMin() const;
        float RadiusMax() const;
        void RepairState(FoldState& s) const;
        bool ContactPoint(const FoldState& s, int a, int b, Vector3* point, int* count) const;
        void ProjectPivot(FoldState& s, int part) const;
    private:
        std::shared_ptr<const ShapeAnalysis> m_Shape;
        FoldParams m_Params;
        std::vector<Vector3> m_Lattice;
        std::vector<DetailSample> m_Detail;
        std::vector<Vector3> m_ShellDirs;
        float m_Spacing = 0.05f;
        float m_DetailScale = 1.0f;
    };

    // Builds the initial hinge tree from contacts between parts (maximum spanning tree, biggest part as root),
    // places the sphere at the centroid and aims every hinge to tuck its subtree into the sphere.
    void BuildFoldTree(FoldState& s, const FoldModel& model);

    // Picks the best axis and angle for one hinge by searching the objective (manual mode "Auto-aim").
    void AimHinge(FoldState& s, const FoldModel& model, int part);

    struct FoldResult {
        bool ok = false;
        bool cancelled = false;
        std::string message;
        Design design;
        FoldMetrics metrics;
        double milliseconds = 0.0;
        int rewritesApplied = 0;
    };

    // Stochastic Rewrite Descent (Kodnongbua et al., "Design for Descent") over the fold grammar: continuous
    // descent on hinge angles / axes / pivots and the sphere, interleaved with batches of grammar rewrites
    // (reparent, split, merge, lock, unlock, axis snap, flip, reseat) whose estimated improvement is positive.
    FoldResult OptimizeFold(const Design& start, std::shared_ptr<const ShapeAnalysis> shape, const FoldParams& params,
                            JobControl& control);
}
