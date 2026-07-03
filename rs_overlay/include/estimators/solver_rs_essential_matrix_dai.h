// Dai et al. (CVPR 2016) RS Essential Matrix Solvers
//
// "Rolling Shutter Camera Relative Pose: Generalized Epipolar Geometry"
//
// The practical implementation follows the reference code:
//   1. N-point SVD for E00 (standard epipolar constraint, ignoring RS)
//   2. Project E00 to essential manifold, decompose to (R, t)
//   3. Linearized RS parameter extraction (omega1, v1, omega2, v2)
//
// Two variants:
//   dai_20pt: Steps 1-3 only (linear RS, 20-point sample)
//   dai_44pt: Steps 1-3 + Gauss-Newton refinement of all 18 params (44-point sample)
//
// Reference: Dai, Li, He. CVPR 2016.

#pragma once

#include <Eigen/Dense>
#include <vector>
#include <cmath>

#include "abstract_solver.h"
#include "rs_utils.h"
#include "../models/model.h"
#include "../utils/types.h"
#include "../utils/macros.h"

namespace superansac {
namespace estimator {
namespace solver {

namespace dai_detail {

// N-point SVD for E00 (standard epipolar, ignoring RS terms).
// Builds an N×9 measurement matrix of q2 ⊗ q1 and takes the last
// right singular vector. Projects to the essential manifold and
// decomposes to (R, t) via cheirality check.
static bool estimateE00(
    const std::vector<Eigen::Vector3d>& q1s,
    const std::vector<Eigen::Vector3d>& q2s,
    int N,
    rs::Pose& pose)
{
    // Build N×9 measurement matrix
    Eigen::MatrixXd A(N, 9);
    for (int i = 0; i < N; i++)
        for (int a = 0; a < 3; a++)
            for (int b = 0; b < 3; b++)
                A(i, 3*a + b) = q2s[i](a) * q1s[i](b);

    // SVD → last right singular vector = E00 (vectorized)
    Eigen::BDCSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::VectorXd e_vec = svd.matrixV().col(8);

    Eigen::Matrix3d E00;
    for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++)
            E00(a, b) = e_vec(3*a + b);

    // Check for degenerate E
    if (!E00.allFinite()) return false;

    // Project to essential matrix manifold
    Eigen::JacobiSVD<Eigen::Matrix3d> svd_E(E00, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d sv_E = svd_E.singularValues();
    double s = (sv_E(0) + sv_E(1)) / 2.0;
    if (s < 1e-15) return false;

    Eigen::Matrix3d E00_proj = svd_E.matrixU() *
        Eigen::Vector3d(s, s, 0).asDiagonal() *
        svd_E.matrixV().transpose();

    // Decompose → (R, t) with cheirality check.
    // rs::decompose_essential transposes E internally (Nister convention).
    // Standard convention: q2^T E q1 = 0, so pass E^T.
    pose = rs::decompose_essential(E00_proj.transpose(), q1s, q2s);
    return true;
}

} // namespace dai_detail


// ============================================================
// Dai 20pt Linear RS Solver
// ============================================================
// Step 1: N-point SVD → E00 (standard epipolar)
// Step 2: Decompose → (R, t)
// Step 3: Linearized RS extraction → (omega, v)

class Dai20ptSolver : public AbstractSolver
{
public:
    Dai20ptSolver(double fy_over_h = 0.7) : fy_over_h_(fy_over_h) {}
    ~Dai20ptSolver() {}

    bool returnMultipleModels() const override { return false; }
    size_t maximumSolutions() const override { return 1; }
    size_t sampleSize() const override { return 20; }

    FORCE_INLINE bool estimateModel(
        const DataMatrix& kData_,
        const size_t *kSample_,
        const size_t kSampleNumber_,
        std::vector<models::Model> &models_,
        const double *kWeights_ = nullptr) const override
    {
        if (kSampleNumber_ < 8) return false;

        const int N = static_cast<int>(kSampleNumber_);
        std::vector<Eigen::Vector3d> q1s, q2s;
        std::vector<double> tau1s, tau2s;
        rs::extract_points_from_data(kData_, kSample_, kSampleNumber_,
                                      q1s, q2s, tau1s, tau2s);

        // Step 1-2: N-point SVD → E00 → decompose
        rs::Pose pose;
        if (!dai_detail::estimateE00(q1s, q2s, N, pose))
            return false;

        // Step 3: Linearized RS extraction
        rs::RSParams rs_params;
        if (N >= 12) {
            rs_params = rs::estimate_rs_params_pc(
                pose.R, pose.t, q1s, q2s, tau1s, tau2s);
        }

        models_.resize(1);
        rs::pack_model(models_[0].getMutableData(), pose.R, pose.t, rs_params);
        return true;
    }

private:
    double fy_over_h_;
};


// ============================================================
// Dai 44pt Full RS Solver
// ============================================================
// Same as 20pt, but with:
//   - Larger sample (44 points) for better conditioning
//   - GN refinement of (R, t) after initial estimation

class Dai44ptSolver : public AbstractSolver
{
public:
    Dai44ptSolver(double fy_over_h = 0.7) : fy_over_h_(fy_over_h) {}
    ~Dai44ptSolver() {}

    bool returnMultipleModels() const override { return false; }
    size_t maximumSolutions() const override { return 1; }
    size_t sampleSize() const override { return 44; }

    FORCE_INLINE bool estimateModel(
        const DataMatrix& kData_,
        const size_t *kSample_,
        const size_t kSampleNumber_,
        std::vector<models::Model> &models_,
        const double *kWeights_ = nullptr) const override
    {
        if (kSampleNumber_ < 8) return false;

        const int N = static_cast<int>(kSampleNumber_);
        std::vector<Eigen::Vector3d> q1s, q2s;
        std::vector<double> tau1s, tau2s;
        rs::extract_points_from_data(kData_, kSample_, kSampleNumber_,
                                      q1s, q2s, tau1s, tau2s);

        // Step 1-2: N-point SVD → E00 → decompose
        rs::Pose pose;
        if (!dai_detail::estimateE00(q1s, q2s, N, pose))
            return false;

        // Step 3: GN refinement of (R, t) using epipolar constraints
        pose = rs::refine_pose_epipolar(pose.R, pose.t, q1s, q2s, 10);

        // Step 4: Linearized RS extraction
        rs::RSParams rs_params;
        if (N >= 12) {
            rs_params = rs::estimate_rs_params_pc(
                pose.R, pose.t, q1s, q2s, tau1s, tau2s);
        }

        models_.resize(1);
        rs::pack_model(models_[0].getMutableData(), pose.R, pose.t, rs_params);
        return true;
    }

private:
    double fy_over_h_;
};

} // namespace solver
} // namespace estimator
} // namespace superansac
