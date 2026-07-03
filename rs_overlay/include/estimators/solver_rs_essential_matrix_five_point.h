// RS Essential Matrix 5-Point Minimal Solver
//
// Pipeline: 5pt Nister -> decompose E -> linearized RS from ACs -> Newton polish
// Sample size: 5 affine correspondences
// Returns: 1 model (R, t, RS params) packed as 8x3 DataMatrix
#pragma once

#include <Eigen/Dense>
#include <vector>

#include "abstract_solver.h"
#include "solver_essential_matrix_five_point_nister.h"
#include "rs_utils.h"
#include "../models/model.h"
#include "../utils/types.h"
#include "../utils/macros.h"

namespace superansac {
namespace estimator {
namespace solver {

class RSEssentialMatrixFivePointSolver : public AbstractSolver
{
protected:
    double fy_over_h_;
    int newton_steps_;

public:
    RSEssentialMatrixFivePointSolver(double fy_over_h, int newton_steps = 3)
        : fy_over_h_(fy_over_h), newton_steps_(newton_steps) {}

    ~RSEssentialMatrixFivePointSolver() {}

    bool returnMultipleModels() const override { return true; }
    size_t maximumSolutions() const override { return 10; }
    size_t sampleSize() const override { return 5; }

    FORCE_INLINE bool estimateModel(
        const DataMatrix& kData_,
        const size_t *kSample_,
        const size_t kSampleNumber_,
        std::vector<models::Model> &models_,
        const double *kWeights_ = nullptr) const override;
};

FORCE_INLINE bool RSEssentialMatrixFivePointSolver::estimateModel(
    const DataMatrix& kData_,
    const size_t *kSample_,
    const size_t kSampleNumber_,
    std::vector<models::Model> &models_,
    const double *kWeights_) const
{
    if (kSampleNumber_ < 5)
        return false;

    // Step 1: Run 5pt Nister on the point parts (cols 0-3)
    EssentialMatrixFivePointNisterSolver fivePointSolver;
    std::vector<models::Model> eModels;
    fivePointSolver.estimateModel(kData_, kSample_, kSampleNumber_, eModels);

    if (eModels.empty())
        return false;

    // Step 2: Extract ACs and point vectors from sample
    std::vector<rs::AffineCorrespondence> acs(kSampleNumber_);
    std::vector<Eigen::Vector3d> q1s(kSampleNumber_), q2s(kSampleNumber_);
    for (size_t i = 0; i < kSampleNumber_; i++) {
        const size_t idx = (kSample_ ? kSample_[i] : i);
        acs[i] = rs::ac_from_data_row(kData_, idx);
        q1s[i] = acs[i].q1;
        q2s[i] = acs[i].q2;
    }

    // Step 3: For each E candidate, decompose, estimate RS, output model
    models_.clear();
    for (auto& eModel : eModels) {
        // Decompose E -> (R, t) with cheirality check
        rs::Pose pose = rs::decompose_essential(eModel.getData().block<3,3>(0,0), q1s, q2s);

        // Refine pose using epipolar GN on all sample points
        rs::Pose refined = rs::refine_pose_epipolar(pose.R, pose.t, acs);

        // Output RS=0 in minimal model. RS estimation from 5 ACs is too
        // noisy and corrupts RS-Sampson scoring during RANSAC.
        // The non-minimal BA solver will estimate RS from the full inlier set.
        rs::RSParams rs_zero;
        models::Model model;
        rs::pack_model(model.getMutableData(), refined.R, refined.t, rs_zero);
        models_.emplace_back(std::move(model));
    }

    return !models_.empty();
}

} // namespace solver
} // namespace estimator
} // namespace superansac
