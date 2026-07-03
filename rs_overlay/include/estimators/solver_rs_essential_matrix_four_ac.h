// RS Essential Matrix 4-AC Minimal Solver
//
// Pipeline:
//   1. From 2 ACs, build 6 linear constraints on E (epipolar + 2 affine each)
//   2. Use 5 constraints with Nister's 5pt method → up to 10 E candidates
//   3. Filter with 6th constraint to select best E
//   4. Decompose E → (R, t) with cheirality check
//   5. Refine pose using epipolar GN on all 4 ACs
//   6. Linearized RS estimation from all 4 ACs (12 eq, 12 unknowns)
//   7. Newton polish on exact RS model
//
// Sample size: 4 affine correspondences
#pragma once

#include <Eigen/Dense>
#include <vector>

#include "solver_essential_matrix_five_point_nister.h"
#include "rs_utils.h"
#include "../models/model.h"
#include "../utils/types.h"
#include "../utils/macros.h"

namespace superansac {
namespace estimator {
namespace solver {

class RSEssentialMatrixFourACSolver : public EssentialMatrixFivePointNisterSolver
{
protected:
    double fy_over_h_;
    int newton_steps_;

public:
    RSEssentialMatrixFourACSolver(double fy_over_h, int newton_steps = 3)
        : EssentialMatrixFivePointNisterSolver(), fy_over_h_(fy_over_h), newton_steps_(newton_steps) {}

    ~RSEssentialMatrixFourACSolver() {}

    bool returnMultipleModels() const override { return true; }
    size_t maximumSolutions() const override { return 10; }
    size_t sampleSize() const override { return 4; }

    FORCE_INLINE bool estimateModel(
        const DataMatrix& kData_,
        const size_t *kSample_,
        const size_t kSampleNumber_,
        std::vector<models::Model> &models_,
        const double *kWeights_ = nullptr) const override;
};

FORCE_INLINE bool RSEssentialMatrixFourACSolver::estimateModel(
    const DataMatrix& kData_,
    const size_t *kSample_,
    const size_t kSampleNumber_,
    std::vector<models::Model> &models_,
    const double *kWeights_) const
{
    if (kSampleNumber_ < 4)
        return false;

    // Step 1: Extract all ACs
    const size_t N = kSampleNumber_;
    std::vector<rs::AffineCorrespondence> acs(N);
    std::vector<Eigen::Vector3d> q1s(N), q2s(N);
    for (size_t i = 0; i < N; i++) {
        const size_t idx = (kSample_ ? kSample_[i] : i);
        acs[i] = rs::ac_from_data_row(kData_, idx);
        q1s[i] = acs[i].q1;
        q2s[i] = acs[i].q2;
    }

    // Step 2: Build 6x9 coefficient matrix from first 2 ACs.
    // Each AC gives 3 linear constraints on E (9 entries):
    //   Epipolar:  q2^T * E * q1 = 0
    //   Affine u:  dq2u^T * E * q1 + q2^T * E * eu = 0,  eu=(1,0,0)
    //   Affine v:  dq2v^T * E * q1 + q2^T * E * ev = 0,  ev=(0,1,0)
    //
    // Coefficient vector for E in row-major order [e00,e01,e02,e10,e11,e12,e20,e21,e22]:
    //   Epipolar row:  [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
    //   Affine u row:  [a*x1+x2, a*y1, a, b*x1+y2, b*y1, b, c*x1+1, c*y1, c]
    //   Affine v row:  [d*x1, d*y1+x2, d, e*x1, e*y1+y2, e, f*x1, f*y1+1, f]
    // where (x1,y1) is image 1 point, (x2,y2) is image 2 point,
    //       (a,b,c) = dq2_dq1(:,0), (d,e,f) = dq2_dq1(:,1)

    using RowMat6x9 = Eigen::Matrix<double, 6, 9, Eigen::RowMajor>;
    RowMat6x9 coefficients;

    for (int ac_idx = 0; ac_idx < 2; ac_idx++) {
        const auto& ac = acs[ac_idx];
        const double x1 = ac.q1(0), y1 = ac.q1(1);
        const double x2 = ac.q2(0), y2 = ac.q2(1);
        const double a = ac.dq2_dq1(0, 0), b = ac.dq2_dq1(1, 0), cv = ac.dq2_dq1(2, 0);
        const double d = ac.dq2_dq1(0, 1), e = ac.dq2_dq1(1, 1), f = ac.dq2_dq1(2, 1);

        const int row_base = ac_idx * 3;

        // Epipolar
        coefficients.row(row_base) <<
            x2*x1, x2*y1, x2,
            y2*x1, y2*y1, y2,
            x1,    y1,    1.0;

        // Affine u: coeff for e_ij = dq2u_i * q1_j + q2_i * delta(j,0)
        coefficients.row(row_base + 1) <<
            a*x1+x2, a*y1,    a,
            b*x1+y2, b*y1,    b,
            cv*x1+1,  cv*y1,   cv;

        // Affine v: coeff for e_ij = dq2v_i * q1_j + q2_i * delta(j,1)
        coefficients.row(row_base + 2) <<
            d*x1,    d*y1+x2, d,
            e*x1,    e*y1+y2, e,
            f*x1,    f*y1+1,  f;
    }

    // Step 3: Use first 5 constraints → 4D null space → Nister solver
    // Save 6th constraint for filtering
    Eigen::Matrix<double, 1, 9> filter_row = coefficients.row(5);

    using RowMat5x9 = Eigen::Matrix<double, 5, 9, Eigen::RowMajor>;
    RowMat5x9 coeff5 = coefficients.topRows<5>();

    const Eigen::FullPivLU<RowMat5x9> lu(coeff5);
    if (lu.dimensionOfKernel() != 4)
        return false;

    Eigen::Matrix<double, 4, 9> nullSpace = lu.kernel().transpose();

    // Step 4: Solve using Nister's trace constraints + polynomial root finding
    std::vector<models::Model> eModels;
    if (!solveFromNullSpace(nullSpace, eModels) || eModels.empty())
        return false;

    // Step 5: Filter E candidates with the 6th constraint
    // Pick the E with smallest residual on the held-out constraint
    int best_idx = 0;
    double best_residual = std::numeric_limits<double>::max();
    for (size_t i = 0; i < eModels.size(); i++) {
        Eigen::Map<const Eigen::Matrix<double, 1, 9>> e_vec(eModels[i].getData().data());
        double residual = std::abs(filter_row.dot(e_vec));
        if (residual < best_residual) {
            best_residual = residual;
            best_idx = static_cast<int>(i);
        }
    }

    // Step 6: Decompose best E → (R, t) with cheirality check
    Eigen::Matrix3d E_best = eModels[best_idx].getData().block<3,3>(0,0);
    rs::Pose pose = rs::decompose_essential(E_best, q1s, q2s);

    // Step 7: Refine pose using epipolar GN on all ACs
    rs::Pose refined = rs::refine_pose_epipolar(pose.R, pose.t, acs);

    // Output RS=0 in minimal model. RS estimation from 4 ACs is too
    // noisy and corrupts RS-Sampson scoring during RANSAC.
    // The non-minimal BA solver will estimate RS from the full inlier set.
    rs::RSParams rs_zero;
    models::Model model;
    rs::pack_model(model.getMutableData(), refined.R, refined.t, rs_zero);
    models_.clear();
    models_.emplace_back(std::move(model));

    return true;
}

} // namespace solver
} // namespace estimator
} // namespace superansac
