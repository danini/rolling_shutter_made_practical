// RS Essential Matrix Non-Minimal Solver (Bundle Adjustment)
//
// Joint Levenberg-Marquardt over pose (R, t) + RS params (omega1, v1, omega2, v2).
// By default, optimizes on epipolar (point) residuals only.
// Set ac_weight > 0 to also use affine frame constraints.
// 17 parameters: rotation(3) + translation(2, tangent plane) + 12 RS params.
#pragma once

#include <Eigen/Dense>
#include <vector>

#include "abstract_solver.h"
#include "solver_rs_essential_matrix_five_point.h"
#include "rs_utils.h"
#include "../models/model.h"
#include "../utils/types.h"
#include "../utils/macros.h"

namespace superansac {
namespace estimator {
namespace solver {

class RSEssentialMatrixBundleAdjustmentSolver : public AbstractSolver
{
protected:
    double fy_over_h_;
    int max_lm_iter_;
    double ac_weight_;

public:
    RSEssentialMatrixBundleAdjustmentSolver(
        double fy_over_h,
        int max_lm_iter = 50,
        double ac_weight = 0.0)
        : fy_over_h_(fy_over_h),
          max_lm_iter_(max_lm_iter),
          ac_weight_(ac_weight) {}

    ~RSEssentialMatrixBundleAdjustmentSolver() {}

    bool returnMultipleModels() const override { return false; }
    size_t maximumSolutions() const override { return 1; }
    size_t sampleSize() const override { return 6; }

    FORCE_INLINE bool estimateModel(
        const DataMatrix& kData_,
        const size_t *kSample_,
        const size_t kSampleNumber_,
        std::vector<models::Model> &models_,
        const double *kWeights_ = nullptr) const override;
};

FORCE_INLINE bool RSEssentialMatrixBundleAdjustmentSolver::estimateModel(
    const DataMatrix& kData_,
    const size_t *kSample_,
    const size_t kSampleNumber_,
    std::vector<models::Model> &models_,
    const double *kWeights_) const
{
    if (kSampleNumber_ < sampleSize())
        return false;

    // If no initial model, estimate from scratch
    if (models_.empty()) {
        RSEssentialMatrixFivePointSolver minSolver(fy_over_h_);
        minSolver.estimateModel(kData_, kSample_, kSampleNumber_, models_);
        if (models_.empty())
            return false;
    }

    // Extract initial model
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    rs::RSParams rs;
    rs::unpack_model(models_[0].getData(), R, t, rs);

    // Build ACs from all sample points
    std::vector<rs::AffineCorrespondence> acs(kSampleNumber_);
    for (size_t i = 0; i < kSampleNumber_; i++) {
        const size_t idx = (kSample_ ? kSample_[i] : i);
        acs[i] = rs::ac_from_data_row(kData_, idx);
    }

    // Iterative refinement: pose-GN → RS-estimate → joint-LM
    // Single outer round matches PCRSRefinementSolver behaviour.
    // More outer iterations hurt accuracy (see MEMORY.md).
    const int N = static_cast<int>(kSampleNumber_);
    const Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();

    rs::JointPoseRS result = {R, t, rs};
    for (int round = 0; round < 1; round++) {
        // Refine pose on ALL points using standard epipolar GN.
        // On first round, uses raw RS-corrupted points.
        // On later rounds, undistorts using current RS estimate.
        if (round == 0) {
            rs::Pose refined = rs::refine_pose_epipolar(R, t, acs);
            R = refined.R;
            t = refined.t;
        } else {
            // Undistort points using current RS, then refine pose
            std::vector<Eigen::Vector3d> q1u(N), q2u(N);
            for (int i = 0; i < N; i++) {
                q1u[i] = rs::undistort_rs(acs[i].q1, acs[i].tau1, result.rs.omega1);
                q2u[i] = rs::undistort_rs(acs[i].q2, acs[i].tau2, result.rs.omega2);
            }
            rs::Pose refined = rs::refine_pose_epipolar(R, t, q1u, q2u);
            R = refined.R;
            t = refined.t;
        }

        // Re-estimate RS from the improved pose using all ACs
        rs::RSParams rs_lin = rs::estimate_rs_params_ac(R, t, acs, fy_over_h_);
        rs = rs::estimate_rs_params_exact_ac(R, t, acs, fy_over_h_, 5, &rs_lin);

        // Joint LM refinement
        if (ac_weight_ > 0) {
            // AC-based: epipolar + affine residuals (17 params)
            result = rs::estimate_joint_pose_rs_ac(
                R, t, acs, fy_over_h_, max_lm_iter_, false, &rs, ac_weight_);
        } else {
            // Epipolar-only (18 params, PC-style)
            std::vector<Eigen::Vector3d> q1v(N), q2v(N);
            std::vector<double> tau1v(N), tau2v(N);
            for (int i = 0; i < N; i++) {
                q1v[i] = acs[i].q1; q2v[i] = acs[i].q2;
                tau1v[i] = acs[i].tau1; tau2v[i] = acs[i].tau2;
            }
            result = rs::estimate_joint_pose_rs(
                R, t, q1v, q2v, tau1v, tau2v, max_lm_iter_, &rs);
        }
        R = result.R;
        t = result.t;
        rs = result.rs;
    }

    // Pack result
    models_.resize(1);
    rs::pack_model(models_[0].getMutableData(), result.R, result.t, result.rs);
    return true;
}

} // namespace solver
} // namespace estimator
} // namespace superansac
