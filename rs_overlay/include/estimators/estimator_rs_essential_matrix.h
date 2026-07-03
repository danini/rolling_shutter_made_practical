// RS Essential Matrix Estimator for SuperANSAC
//
// Estimates relative pose + rolling shutter parameters from affine correspondences.
// Data format: each row = [x1, y1, x2, y2, tau1, tau2, a00, a10, a20, a01, a11, a21]
// Model format: 8x3 matrix [R; t^T; omega1^T; v1^T; omega2^T; v2^T]
//
// Minimal solver: 5pt + linearized RS + Newton (5 ACs)
// Non-minimal solver: Joint LM over pose + RS (17 params)
// Residual: RS-aware Sampson distance
#pragma once

#define _USE_MATH_DEFINES

#include <math.h>
#include <cmath>
#include <limits>
#include <vector>

#include <Eigen/Eigen>

#include "abstract_estimator.h"
#include "rs_utils.h"
#include "../models/model.h"
#include "../utils/types.h"

namespace superansac {
namespace estimator {

class RSEssentialMatrixEstimator : public Estimator
{
protected:
    double fy_over_h_;

    // Per-hypothesis E-matrix cache (mutable so squaredResidual can update it).
    // The scorer calls squaredResidual(point, model) N times for one hypothesis.
    // Rather than recomputing E = skew(t)*R on every call, we cache it and
    // detect hypothesis changes by comparing t(0) (a double; exact equality is
    // safe because the same DataMatrix object is reused for all N points of the
    // same hypothesis).  Each pybind call creates a fresh estimator so this is
    // thread-safe across concurrent RANSAC workers.
    //
    // NOTE: We use a bool flag + double key instead of NaN-based invalidation
    // because -ffast-math makes NaN comparisons unreliable.
    mutable bool            e_cache_valid_ = false;
    mutable double          e_cache_key_   = 0.0;
    mutable Eigen::Matrix3d e_cache_       = Eigen::Matrix3d::Zero();

public:
    RSEssentialMatrixEstimator(double fy_over_h)
        : fy_over_h_(fy_over_h)
    {
    }

    ~RSEssentialMatrixEstimator() {}

    bool isWeightingApplicable() const override { return true; }

    double multError() const { return 1.0; }

    double logAlpha0(size_t w, size_t h, double scalingFactor = 0.5) const
    {
        return log(1.0 / (w * h * scalingFactor));
    }

    size_t getDegreesOfFreedom() const { return 2; }

    // Estimating the model from a minimal sample
    FORCE_INLINE bool estimateModel(
        const DataMatrix& kData_,
        const size_t *kSample_,
        std::vector<models::Model>* models_) const override
    {
        const size_t kSampleSize = sampleSize();

        minimalSolver->estimateModel(kData_,
            kSample_,
            kSampleSize,
            *models_);

        return models_->size() > 0;
    }

    // Estimating the model from a non-minimal sample
    FORCE_INLINE bool estimateModelNonminimal(
        const DataMatrix& kData_,
        const size_t *kSample_,
        const size_t &kSampleNumber_,
        std::vector<models::Model>* models_,
        const double *kWeights_ = nullptr) const override
    {
        if (kSampleNumber_ < nonMinimalSampleSize())
            return false;

        if (!nonMinimalSolver->estimateModel(kData_,
            kSample_,
            kSampleNumber_,
            *models_,
            kWeights_))
            return false;
        return true;
    }

    FORCE_INLINE double squaredResidual(const DataMatrix& point_,
        const models::Model& model_) const override
    {
        return squaredResidual(point_, model_.getData());
    }

    // RS-aware Sampson distance.
    //
    // Fast path: the 5pt minimal solver always outputs RS=0 (omega=v=0).
    // In that case R_tilde=R and t_tilde=t, so we skip the per-AC matrix
    // multiplications and precompute E=skew(t)*R once.  The branch predictor
    // will learn "always take fast path" during RANSAC, making the check
    // essentially free (~12 FLOPs saved ~150 FLOPs of matrix work per AC).
    FORCE_INLINE double squaredResidual(
        const DataMatrix& kPoint_,
        const DataMatrix& kDescriptor_) const
    {
        // Extract model (8x3)
        const Eigen::Matrix3d R = kDescriptor_.block<3,3>(0,0);
        const Eigen::Vector3d t = kDescriptor_.row(3).transpose();
        const Eigen::Vector3d omega1 = kDescriptor_.row(4).transpose();
        const Eigen::Vector3d v1 = kDescriptor_.row(5).transpose();
        const Eigen::Vector3d omega2 = kDescriptor_.row(6).transpose();
        const Eigen::Vector3d v2 = kDescriptor_.row(7).transpose();

        // Extract point data (1×12 row)
        const double x1 = kPoint_(0, 0), y1 = kPoint_(0, 1);
        const double x2 = kPoint_(0, 2), y2 = kPoint_(0, 3);
        const double tau1 = kPoint_(0, 4), tau2 = kPoint_(0, 5);

        Eigen::Matrix3d E;
        if ((omega1.squaredNorm() + omega2.squaredNorm()
             + v1.squaredNorm() + v2.squaredNorm()) < 1e-30)
        {
            // RS=0 fast path (always taken during RANSAC since the 5pt minimal
            // solver outputs zero RS).  E = skew(t)*R is constant for the
            // entire hypothesis.  Cache it: the scorer calls squaredResidual N
            // times with the same model — we only compute E once per hypothesis.
            if (!e_cache_valid_ || t(0) != e_cache_key_) {
                e_cache_       = rs::skew(t) * R;
                e_cache_key_   = t(0);
                e_cache_valid_ = true;
            }
            E = e_cache_;
        }
        else
        {
            // Full RS path: row-dependent essential matrix (no caching here as
            // E varies per point through tau1, tau2)
            const Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();
            const Eigen::Matrix3d R_tilde = (I3 + tau2 * rs::skew(omega2)) * R *
                                            (I3 - tau1 * rs::skew(omega1));
            const Eigen::Vector3d t_tilde = (t + tau2 * v2) - tau1 * R_tilde * v1;
            E = rs::skew(t_tilde) * R_tilde;
        }

        // Standard Sampson distance with E (or E_tilde)
        const double E0_0 = E(0,0), E0_1 = E(0,1), E0_2 = E(0,2);
        const double E1_0 = E(1,0), E1_1 = E(1,1), E1_2 = E(1,2);
        const double E2_0 = E(2,0), E2_1 = E(2,1), E2_2 = E(2,2);

        const double Ex1_0 = E0_0 * x1 + E0_1 * y1 + E0_2;
        const double Ex1_1 = E1_0 * x1 + E1_1 * y1 + E1_2;
        const double Ex1_2 = E2_0 * x1 + E2_1 * y1 + E2_2;

        const double Ex2_0 = E0_0 * x2 + E1_0 * y2 + E2_0;
        const double Ex2_1 = E0_1 * x2 + E1_1 * y2 + E2_1;

        const double C = x2 * Ex1_0 + y2 * Ex1_1 + Ex1_2;
        const double Cx = Ex1_0 * Ex1_0 + Ex1_1 * Ex1_1;
        const double Cy = Ex2_0 * Ex2_0 + Ex2_1 * Ex2_1;

        const double denom_sum = Cx + Cy;
        if (denom_sum < 1e-30)
            return C * C; // degenerate, return algebraic error

        return (C * C) / denom_sum;
    }

    FORCE_INLINE double residual(const DataMatrix& point_,
        const models::Model& model_) const override
    {
        return residual(point_, model_.getData());
    }

    FORCE_INLINE double residual(
        const DataMatrix& point_,
        const DataMatrix& descriptor_) const
    {
        return sqrt(squaredResidual(point_, descriptor_));
    }

    FORCE_INLINE bool isValidModel(models::Model& model_,
        const DataMatrix& kData_,
        const size_t* kMinimalSample_,
        const double kThreshold_,
        bool& modelUpdated_) const override
    {
        return true;
    }

    FORCE_INLINE bool isValidSample(
        const DataMatrix& kData_,
        const size_t *kSample_) const override
    {
        return true;
    }
};

} // namespace estimator
} // namespace superansac
