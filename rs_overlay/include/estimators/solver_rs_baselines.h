// Baseline RS Essential Matrix Solvers for SuperANSAC
//
// Provides minimal and non-minimal solvers for all baseline methods:
//   GS-5pt, PC-RS, Npt+linRS, Npt+linRS+GN, Npt+cvRS, Npt+iterRS
//
// All produce the standard 8x3 model format [R; t^T; omega1^T; v1^T; omega2^T; v2^T].
// Data format: [x1, y1, x2, y2, tau1, tau2, ...] (columns 6+ ignored by PC methods).
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

// ============================================================
// Minimal: GS 5-point (no RS)
// ============================================================
// Nister 5pt → decompose E → pack with RS = 0.

class GSFivePointSolver : public AbstractSolver
{
public:
    GSFivePointSolver() {}
    ~GSFivePointSolver() {}

    bool returnMultipleModels() const override { return true; }
    size_t maximumSolutions() const override { return 10; }
    size_t sampleSize() const override { return 5; }

    FORCE_INLINE bool estimateModel(
        const DataMatrix& kData_,
        const size_t *kSample_,
        const size_t kSampleNumber_,
        std::vector<models::Model> &models_,
        const double *kWeights_ = nullptr) const override
    {
        if (kSampleNumber_ < 5) return false;

        EssentialMatrixFivePointNisterSolver fivePointSolver;
        std::vector<models::Model> eModels;
        fivePointSolver.estimateModel(kData_, kSample_, kSampleNumber_, eModels);
        if (eModels.empty()) return false;

        std::vector<Eigen::Vector3d> q1s(kSampleNumber_), q2s(kSampleNumber_);
        for (size_t i = 0; i < kSampleNumber_; i++) {
            const size_t idx = kSample_ ? kSample_[i] : i;
            q1s[i] = Eigen::Vector3d(kData_(idx, 0), kData_(idx, 1), 1.0);
            q2s[i] = Eigen::Vector3d(kData_(idx, 2), kData_(idx, 3), 1.0);
        }

        models_.clear();
        for (auto& eModel : eModels) {
            rs::Pose pose = rs::decompose_essential(
                eModel.getData().block<3,3>(0,0), q1s, q2s);

            models::Model model;
            rs::RSParams rs_zero;
            rs::pack_model(model.getMutableData(), pose.R, pose.t, rs_zero);
            models_.emplace_back(std::move(model));
        }
        return !models_.empty();
    }
};

// ============================================================
// Minimal: PC 5-point + RS extraction (no affine info)
// ============================================================
// 5pt → decompose E → PC-based linearized RS → short joint LM.

class PCFivePointRSSolver : public AbstractSolver
{
public:
    PCFivePointRSSolver() {}
    ~PCFivePointRSSolver() {}

    bool returnMultipleModels() const override { return true; }
    size_t maximumSolutions() const override { return 10; }
    size_t sampleSize() const override { return 5; }

    FORCE_INLINE bool estimateModel(
        const DataMatrix& kData_,
        const size_t *kSample_,
        const size_t kSampleNumber_,
        std::vector<models::Model> &models_,
        const double *kWeights_ = nullptr) const override
    {
        if (kSampleNumber_ < 5) return false;

        EssentialMatrixFivePointNisterSolver fivePointSolver;
        std::vector<models::Model> eModels;
        fivePointSolver.estimateModel(kData_, kSample_, kSampleNumber_, eModels);
        if (eModels.empty()) return false;

        std::vector<Eigen::Vector3d> q1s, q2s;
        std::vector<double> tau1s, tau2s;
        rs::extract_points_from_data(kData_, kSample_, kSampleNumber_,
                                      q1s, q2s, tau1s, tau2s);

        models_.clear();
        for (auto& eModel : eModels) {
            rs::Pose pose = rs::decompose_essential(
                eModel.getData().block<3,3>(0,0), q1s, q2s);

            // Refine pose with epipolar GN
            rs::Pose refined = rs::refine_pose_epipolar(pose.R, pose.t, q1s, q2s, 10);

            // PC-based RS extraction (needs >= 12 points; with 5, use zero init)
            rs::RSParams rs;
            if (kSampleNumber_ >= 12) {
                rs = rs::estimate_rs_params_pc(
                    refined.R, refined.t, q1s, q2s, tau1s, tau2s);
                // Short joint LM
                rs::JointPoseRS result = rs::estimate_joint_pose_rs(
                    refined.R, refined.t, q1s, q2s, tau1s, tau2s, 5, &rs);
                refined.R = result.R;
                refined.t = result.t;
                rs = result.rs;
            }

            models::Model model;
            rs::pack_model(model.getMutableData(), refined.R, refined.t, rs);
            models_.emplace_back(std::move(model));
        }
        return !models_.empty();
    }
};

// ============================================================
// Minimal: 8-point essential matrix solver
// ============================================================
// N-point SVD → essential matrix projection → decompose.
// RS estimated via linearized PC extraction.

class EightPointEssentialSolver : public AbstractSolver
{
public:
    EightPointEssentialSolver() {}
    ~EightPointEssentialSolver() {}

    bool returnMultipleModels() const override { return false; }
    size_t maximumSolutions() const override { return 1; }
    size_t sampleSize() const override { return 8; }

    FORCE_INLINE bool estimateModel(
        const DataMatrix& kData_,
        const size_t *kSample_,
        const size_t kSampleNumber_,
        std::vector<models::Model> &models_,
        const double *kWeights_ = nullptr) const override
    {
        if (kSampleNumber_ < 8) return false;

        std::vector<Eigen::Vector3d> q1s, q2s;
        std::vector<double> tau1s, tau2s;
        rs::extract_points_from_data(kData_, kSample_, kSampleNumber_,
                                      q1s, q2s, tau1s, tau2s);

        rs::Pose pose = rs::solve_8point_essential(q1s, q2s);

        // Linearized RS extraction if enough points
        rs::RSParams rs;
        if (kSampleNumber_ >= 12) {
            rs = rs::estimate_rs_params_pc(pose.R, pose.t, q1s, q2s, tau1s, tau2s);
        }

        models_.resize(1);
        rs::pack_model(models_[0].getMutableData(), pose.R, pose.t, rs);
        return true;
    }
};

// ============================================================
// Non-minimal: GS refinement (pose-only GN, no RS)
// ============================================================

class GSRefinementSolver : public AbstractSolver
{
public:
    GSRefinementSolver() {}
    ~GSRefinementSolver() {}

    bool returnMultipleModels() const override { return false; }
    size_t maximumSolutions() const override { return 1; }
    size_t sampleSize() const override { return 6; }

    FORCE_INLINE bool estimateModel(
        const DataMatrix& kData_,
        const size_t *kSample_,
        const size_t kSampleNumber_,
        std::vector<models::Model> &models_,
        const double *kWeights_ = nullptr) const override
    {
        if (kSampleNumber_ < sampleSize()) return false;

        // Need an initial model
        if (models_.empty()) {
            GSFivePointSolver minSolver;
            minSolver.estimateModel(kData_, kSample_, kSampleNumber_, models_);
            if (models_.empty()) return false;
        }

        Eigen::Matrix3d R;
        Eigen::Vector3d t;
        rs::RSParams rs;
        rs::unpack_model(models_[0].getData(), R, t, rs);

        std::vector<Eigen::Vector3d> q1s, q2s;
        std::vector<double> tau1s, tau2s;
        rs::extract_points_from_data(kData_, kSample_, kSampleNumber_,
                                      q1s, q2s, tau1s, tau2s);

        // Pose-only GN
        rs::Pose refined = rs::refine_pose_epipolar(R, t, q1s, q2s, 50);

        models_.resize(1);
        rs::RSParams rs_zero;
        rs::pack_model(models_[0].getMutableData(), refined.R, refined.t, rs_zero);
        return true;
    }
};

// ============================================================
// Non-minimal: PC-RS refinement (joint LM, iterative)
// ============================================================
// Same iterative structure as RSEssentialMatrixBundleAdjustmentSolver
// but uses PC-based functions instead of AC-based ones.

class PCRSRefinementSolver : public AbstractSolver
{
public:
    PCRSRefinementSolver() {}
    ~PCRSRefinementSolver() {}

    bool returnMultipleModels() const override { return false; }
    size_t maximumSolutions() const override { return 1; }
    size_t sampleSize() const override { return 6; }

    FORCE_INLINE bool estimateModel(
        const DataMatrix& kData_,
        const size_t *kSample_,
        const size_t kSampleNumber_,
        std::vector<models::Model> &models_,
        const double *kWeights_ = nullptr) const override
    {
        if (kSampleNumber_ < sampleSize()) return false;

        if (models_.empty()) {
            PCFivePointRSSolver minSolver;
            minSolver.estimateModel(kData_, kSample_, kSampleNumber_, models_);
            if (models_.empty()) return false;
        }

        Eigen::Matrix3d R;
        Eigen::Vector3d t;
        rs::RSParams rs;
        rs::unpack_model(models_[0].getData(), R, t, rs);

        int N = static_cast<int>(kSampleNumber_);
        std::vector<Eigen::Vector3d> q1s, q2s;
        std::vector<double> tau1s, tau2s;
        rs::extract_points_from_data(kData_, kSample_, kSampleNumber_,
                                      q1s, q2s, tau1s, tau2s);

        // Iterative: undistort → pose GN → RS extract → joint LM
        for (int round = 0; round < 1; round++) {
            if (round == 0) {
                rs::Pose refined = rs::refine_pose_epipolar(R, t, q1s, q2s);
                R = refined.R;
                t = refined.t;
            } else {
                std::vector<Eigen::Vector3d> q1u(N), q2u(N);
                for (int i = 0; i < N; i++) {
                    q1u[i] = rs::undistort_rs(q1s[i], tau1s[i], rs.omega1);
                    q2u[i] = rs::undistort_rs(q2s[i], tau2s[i], rs.omega2);
                }
                rs::Pose refined = rs::refine_pose_epipolar(R, t, q1u, q2u);
                R = refined.R;
                t = refined.t;
            }

            if (N >= 12) {
                rs = rs::estimate_rs_params_pc(R, t, q1s, q2s, tau1s, tau2s);
            }

            rs::JointPoseRS result = rs::estimate_joint_pose_rs(
                R, t, q1s, q2s, tau1s, tau2s, 50, &rs);
            R = result.R;
            t = result.t;
            rs = result.rs;
        }

        models_.resize(1);
        rs::pack_model(models_[0].getMutableData(), R, t, rs);
        return true;
    }
};

// ============================================================
// Non-minimal: Linear RS only (dai_lin style)
// ============================================================
// Refine pose → extract linearized RS params. No joint LM.

class LinearRSRefinementSolver : public AbstractSolver
{
public:
    LinearRSRefinementSolver() {}
    ~LinearRSRefinementSolver() {}

    bool returnMultipleModels() const override { return false; }
    size_t maximumSolutions() const override { return 1; }
    size_t sampleSize() const override { return 12; }

    FORCE_INLINE bool estimateModel(
        const DataMatrix& kData_,
        const size_t *kSample_,
        const size_t kSampleNumber_,
        std::vector<models::Model> &models_,
        const double *kWeights_ = nullptr) const override
    {
        if (kSampleNumber_ < sampleSize()) return false;

        if (models_.empty()) {
            EightPointEssentialSolver minSolver;
            minSolver.estimateModel(kData_, kSample_, kSampleNumber_, models_);
            if (models_.empty()) return false;
        }

        Eigen::Matrix3d R;
        Eigen::Vector3d t;
        rs::RSParams rs;
        rs::unpack_model(models_[0].getData(), R, t, rs);

        std::vector<Eigen::Vector3d> q1s, q2s;
        std::vector<double> tau1s, tau2s;
        rs::extract_points_from_data(kData_, kSample_, kSampleNumber_,
                                      q1s, q2s, tau1s, tau2s);

        // Pose-only GN then linearized RS
        rs::Pose refined = rs::refine_pose_epipolar(R, t, q1s, q2s, 20);
        rs = rs::estimate_rs_params_pc(refined.R, refined.t, q1s, q2s, tau1s, tau2s);

        models_.resize(1);
        rs::pack_model(models_[0].getMutableData(), refined.R, refined.t, rs);
        return true;
    }
};

// ============================================================
// Non-minimal: Linear RS + GN refinement (dai_gn style)
// ============================================================

class GNRSRefinementSolver : public AbstractSolver
{
public:
    GNRSRefinementSolver() {}
    ~GNRSRefinementSolver() {}

    bool returnMultipleModels() const override { return false; }
    size_t maximumSolutions() const override { return 1; }
    size_t sampleSize() const override { return 12; }

    FORCE_INLINE bool estimateModel(
        const DataMatrix& kData_,
        const size_t *kSample_,
        const size_t kSampleNumber_,
        std::vector<models::Model> &models_,
        const double *kWeights_ = nullptr) const override
    {
        if (kSampleNumber_ < sampleSize()) return false;

        if (models_.empty()) {
            EightPointEssentialSolver minSolver;
            minSolver.estimateModel(kData_, kSample_, kSampleNumber_, models_);
            if (models_.empty()) return false;
        }

        Eigen::Matrix3d R;
        Eigen::Vector3d t;
        rs::RSParams rs;
        rs::unpack_model(models_[0].getData(), R, t, rs);

        std::vector<Eigen::Vector3d> q1s, q2s;
        std::vector<double> tau1s, tau2s;
        rs::extract_points_from_data(kData_, kSample_, kSampleNumber_,
                                      q1s, q2s, tau1s, tau2s);

        // Pose GN → linearized RS → joint LM
        rs::Pose refined = rs::refine_pose_epipolar(R, t, q1s, q2s, 20);
        rs = rs::estimate_rs_params_pc(refined.R, refined.t, q1s, q2s, tau1s, tau2s);
        rs::JointPoseRS result = rs::estimate_joint_pose_rs(
            refined.R, refined.t, q1s, q2s, tau1s, tau2s, 20, &rs);

        models_.resize(1);
        rs::pack_model(models_[0].getMutableData(), result.R, result.t, result.rs);
        return true;
    }
};

// ============================================================
// Non-minimal: Constant-velocity RS (cv_rs style)
// ============================================================

class CVRSRefinementSolver : public AbstractSolver
{
public:
    CVRSRefinementSolver() {}
    ~CVRSRefinementSolver() {}

    bool returnMultipleModels() const override { return false; }
    size_t maximumSolutions() const override { return 1; }
    size_t sampleSize() const override { return 8; }

    FORCE_INLINE bool estimateModel(
        const DataMatrix& kData_,
        const size_t *kSample_,
        const size_t kSampleNumber_,
        std::vector<models::Model> &models_,
        const double *kWeights_ = nullptr) const override
    {
        if (kSampleNumber_ < sampleSize()) return false;

        if (models_.empty()) {
            EightPointEssentialSolver minSolver;
            minSolver.estimateModel(kData_, kSample_, kSampleNumber_, models_);
            if (models_.empty()) return false;
        }

        Eigen::Matrix3d R;
        Eigen::Vector3d t;
        rs::RSParams rs;
        rs::unpack_model(models_[0].getData(), R, t, rs);

        std::vector<Eigen::Vector3d> q1s, q2s;
        std::vector<double> tau1s, tau2s;
        rs::extract_points_from_data(kData_, kSample_, kSampleNumber_,
                                      q1s, q2s, tau1s, tau2s);

        rs::Pose refined = rs::refine_pose_epipolar(R, t, q1s, q2s, 20);
        rs = rs::estimate_rs_params_cv(refined.R, refined.t, q1s, q2s, tau1s, tau2s);

        models_.resize(1);
        rs::pack_model(models_[0].getMutableData(), refined.R, refined.t, rs);
        return true;
    }
};

// ============================================================
// Non-minimal: Iterative RS undistortion (iter_rs style)
// ============================================================
// Loop: 8pt on corrected points → RS extraction → undistort → repeat.

class IterativeRSRefinementSolver : public AbstractSolver
{
    int max_iters_;
public:
    IterativeRSRefinementSolver(int max_iters = 3) : max_iters_(max_iters) {}
    ~IterativeRSRefinementSolver() {}

    bool returnMultipleModels() const override { return false; }
    size_t maximumSolutions() const override { return 1; }
    size_t sampleSize() const override { return 12; }

    FORCE_INLINE bool estimateModel(
        const DataMatrix& kData_,
        const size_t *kSample_,
        const size_t kSampleNumber_,
        std::vector<models::Model> &models_,
        const double *kWeights_ = nullptr) const override
    {
        if (kSampleNumber_ < sampleSize()) return false;

        int N = static_cast<int>(kSampleNumber_);
        std::vector<Eigen::Vector3d> q1s, q2s;
        std::vector<double> tau1s, tau2s;
        rs::extract_points_from_data(kData_, kSample_, kSampleNumber_,
                                      q1s, q2s, tau1s, tau2s);

        // Working copies
        std::vector<Eigen::Vector3d> q1w = q1s, q2w = q2s;
        Eigen::Matrix3d R;
        Eigen::Vector3d t;
        rs::RSParams rs;

        for (int iter = 0; iter < max_iters_; iter++) {
            // Solve E from (possibly corrected) correspondences
            rs::Pose pose = rs::solve_8point_essential(q1w, q2w);
            R = pose.R;
            t = pose.t;

            // Extract RS using ORIGINAL correspondences
            rs = rs::estimate_rs_params_pc(R, t, q1s, q2s, tau1s, tau2s);

            // Undistort for next iteration
            if (iter < max_iters_ - 1) {
                for (int i = 0; i < N; i++) {
                    q1w[i] = rs::undistort_rs(q1s[i], tau1s[i], rs.omega1);
                    q2w[i] = rs::undistort_rs(q2s[i], tau2s[i], rs.omega2);
                }
            }
        }

        models_.resize(1);
        rs::pack_model(models_[0].getMutableData(), R, t, rs);
        return true;
    }
};

} // namespace solver
} // namespace estimator
} // namespace superansac
