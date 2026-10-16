#pragma once
#include "data/job.hpp"
#include "data/shape.hpp"
#include "data/fitloss.hpp"
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
        bool checkPath = true;           // the motion between figure and ball must be collision free
        bool chained = true;             // a single latch: hinges release one another in a chain
        bool joint = false;              // optimise the superfrustums together with the fold
        float wCoverage = 2.5f;          // sphere shell left uncovered
        float wShellOverlap = 0.6f;      // shell covered twice
        float wTrim = 1.0f;              // material cut off by the sphere
        float wCollision = 2.5f;         // material of different parts in the same place when closed
        float wPath = 2.0f;              // the finished pieces passing through each other while folding / unfolding
        float wDetail = 1.0f;            // extra weight of surface detail (curvature) in the trim term
        float wContact = 0.5f;           // hinge pivots must sit inside both parts
        float wVolume = 0.4f;            // keep the sphere near the equal-volume size (within the tolerance band)
        float wLink = 1.0f;              // hinge pivots must stay inside the closed sphere, or the chain breaks
        float wFit = 25.0f;              // joint mode: how faithfully the superfrustums must keep reconstructing the input
        float hingeCost = 0.004f;        // simplicity: cost per working hinge
        int pathSamples = 5;             // moments of the motion checked during optimisation (validation uses more)
        int shapeBlock = 1;              // joint mode: superfrustums updated per descent step
        int samplesPerPart = 260;
        int shellSamples = 640;
        int maxParts = 24;
        unsigned seed = 7;

        bool operator==(const FoldParams& o) const {
            return volumeTolerance == o.volumeTolerance && maxHingeAngle == o.maxHingeAngle && iterations == o.iterations &&
                   rewriteEvery == o.rewriteEvery && candidates == o.candidates && restructure == o.restructure &&
                   keepTree == o.keepTree && allowSplit == o.allowSplit && checkPath == o.checkPath && chained == o.chained &&
                   joint == o.joint && wCoverage == o.wCoverage && wShellOverlap == o.wShellOverlap && wTrim == o.wTrim &&
                   wCollision == o.wCollision && wPath == o.wPath && wDetail == o.wDetail && wContact == o.wContact &&
                   wVolume == o.wVolume && wLink == o.wLink && wFit == o.wFit && hingeCost == o.hingeCost &&
                   pathSamples == o.pathSamples && shapeBlock == o.shapeBlock && samplesPerPart == o.samplesPerPart &&
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
        float pathCollision = 0;         // worst share of the pieces in contact at any moment of the motion
        float pathWorstT = 0;            // fold fraction where that happens
        bool foldable = false;           // the motion is collision free (within tolerance)
        float contact = 0;               // hinge pivots outside their parts (relative)
        float fitIoU = -1;               // superfrustums vs. input (when known)
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
        // ownership cache: distance of every material sample (lattice, then detail) to every superfrustum
        std::vector<float> distance;
        int columns = 0;
        float fitLoss = 0.0f;               // joint mode: reconstruction loss of the superfrustums
    };

    // The physical pieces as point sets: surviving material plus the shell plates, each owned by a part (open pose).
    struct PieceSamples {
        std::vector<Vector3> points;
        std::vector<int> owner;
        std::vector<uint8_t> plate;          // a shell plate sample (not input material)
        std::vector<uint64_t> resting;       // sorted pairs already touching in the open pose (the cut between parts)
        float openClash = 0.0f;              // share of plate samples buried in another part's material in the figure
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
        const FitModel& Fit() const { return m_Fit; }

        FoldState MakeState(const Design& d) const;
        void Assign(FoldState& s) const;
        void UpdateFit(FoldState& s) const;
        float FitIoU(const Design& d) const;
        // Fold objective (no fit term) of a design with the given ownership.
        float Evaluate(const Design& d, const std::vector<PartSamples>& samples, FoldMetrics* metrics = nullptr,
                       FoldVisual* visual = nullptr) const;
        // Full objective of a state (adds the fit term in joint mode).
        float Evaluate(const FoldState& s, FoldMetrics* metrics = nullptr, FoldVisual* visual = nullptr) const;
        // Fold objective with part `i`'s superfrustum replaced and the material ownership updated accordingly.
        float EvaluateShape(const FoldState& s, int i, const SuperFrustum& sf) const;
        float CollisionLoss(const std::vector<PartSamples>& samples, const std::vector<Rigid>& xf, float* fraction,
                            std::vector<Vector3>* lost, int stride = 1) const;
        // Shifts the moments of the motion sampled during optimisation (set between steps, not during evaluation).
        void SetMotionPhase(float phase) { m_Phase = phase; }
        void SetMotionBoost(float boost) { m_Boost = boost; }   // multiplies the motion term (repair phase)
        // Pieces in contact at fold fraction t (for the viewport).
        void MotionContacts(const FoldState& s, float t, std::vector<Vector3>* points) const;
        float RadiusMin() const;
        float RadiusMax() const;
        void RepairState(FoldState& s) const;
        bool ContactPoint(const FoldState& s, int a, int b, Vector3* point, int* count) const;
        void ProjectPivot(FoldState& s, int part) const;
    private:
        void BuildPieces(const Design& d, const std::vector<PartSamples>& samples, const std::vector<Rigid>& closed,
                         const std::vector<std::vector<Vector3>>& plates, PieceSamples& out) const;
        float MotionLoss(const Design& d, const PieceSamples& pieces, const std::vector<float>& times, int stride,
                         float* worst, float* worstT, std::vector<Vector3>* contacts) const;
        std::shared_ptr<const ShapeAnalysis> m_Shape;
        FoldParams m_Params;
        std::vector<Vector3> m_Lattice;
        std::vector<DetailSample> m_Detail;
        std::vector<Vector3> m_ShellDirs;
        FitModel m_Fit;
        float m_Spacing = 0.05f;
        float m_DetailScale = 1.0f;
        float m_Phase = 0.5f;
        float m_Boost = 1.0f;
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
    // descent on hinge angles / axes / pivots / release points and the sphere (and, in joint mode, the superfrustums),
    // interleaved with batches of grammar rewrites whose estimated improvement is positive.
    FoldResult OptimizeFold(const Design& start, std::shared_ptr<const ShapeAnalysis> shape, const FoldParams& params,
                            JobControl& control);
}
