// RS Relative Pose Utilities
// Ported from solve_rs_iterative.cpp for use in the SuperANSAC framework.
//
// RS camera model (linearized):
//   Camera 1 (reference): R1(y) = I + tau1*[w1]x,  t1(y) = tau1*v1
//   Camera 2:             R2(y) = (I + tau2*[w2]x)*R0,  t2(y) = R0*t + tau2*v2
//   where tau_i = (y_i - h/2) / h  (normalized row coordinate, in [-0.5, 0.5])
//
//   Per-row relative pose:
//     R_tilde = (I + tau2*[w2]x) * R0 * (I - tau1*[w1]x)
//     t_tilde = (t + tau2*v2) - tau1*R_tilde*v1
//     E_tilde = [t_tilde]x * R_tilde
#pragma once

#include <Eigen/Dense>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <utility>

namespace superansac {
namespace estimator {
namespace rs {

using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::RowVector4d;
using Eigen::Matrix;
using Eigen::JacobiSVD;
using Eigen::ComputeFullU;
using Eigen::ComputeFullV;

// ============================================================
// Basic utilities
// ============================================================

inline Matrix3d skew(const Vector3d& v) {
    Matrix3d S;
    S <<     0, -v(2),  v(1),
          v(2),     0, -v(0),
         -v(1),  v(0),     0;
    return S;
}

inline Matrix3d axis_angle_to_rot(const Vector3d& aa) {
    double angle = aa.norm();
    if (angle < 1e-12) return Matrix3d::Identity();
    Vector3d axis = aa / angle;
    Matrix3d K = skew(axis);
    return Matrix3d::Identity() + std::sin(angle) * K + (1 - std::cos(angle)) * K * K;
}

// ============================================================
// RS data structures
// ============================================================

struct RSParams {
    Vector3d omega1, v1, omega2, v2;
    RSParams() : omega1(Vector3d::Zero()), v1(Vector3d::Zero()),
                 omega2(Vector3d::Zero()), v2(Vector3d::Zero()) {}
    RSParams(const Vector3d& w1, const Vector3d& v1_,
             const Vector3d& w2, const Vector3d& v2_)
        : omega1(w1), v1(v1_), omega2(w2), v2(v2_) {}
};

struct AffineCorrespondence {
    Vector3d q1, q2;
    double tau1, tau2;
    Matrix<double, 3, 2> dq2_dq1;
};

struct Pose {
    Matrix3d R;
    Vector3d t;
};

struct JointPoseRS {
    Matrix3d R;
    Vector3d t;
    RSParams rs;
};

// ============================================================
// RS essential matrix computation
// ============================================================

inline Matrix3d rs_essential(const Matrix3d& R0, const Vector3d& t,
                             const RSParams& rs, double tau1, double tau2)
{
    Matrix3d R_tilde = (Matrix3d::Identity() + tau2 * skew(rs.omega2)) * R0 *
                       (Matrix3d::Identity() - tau1 * skew(rs.omega1));
    Vector3d t_tilde = (t + tau2 * rs.v2) - tau1 * R_tilde * rs.v1;
    return skew(t_tilde) * R_tilde;
}

inline Vector3d undistort_rs(const Vector3d& q, double tau, const Vector3d& omega) {
    return q - tau * omega.cross(q);
}

// ============================================================
// E decomposition with cheirality check
// ============================================================
// Note: the 5pt solver in superansac returns E such that vec(E) satisfies
// the Kronecker product constraint, with q1^T E q2 = 0.
// For standard q2^T F q1 = 0, we need F = E^T.

inline Pose decompose_essential(const Matrix3d& E_in,
                                const std::vector<Vector3d>& q1,
                                const std::vector<Vector3d>& q2)
{
    Matrix3d E = E_in.transpose();

    JacobiSVD<Matrix3d> svd(E, ComputeFullU | ComputeFullV);
    Matrix3d U = svd.matrixU();
    Matrix3d V = svd.matrixV();

    if (U.determinant() < 0) U.col(2) *= -1;
    if (V.determinant() < 0) V.col(2) *= -1;

    Matrix3d W;
    W << 0, -1, 0,
         1,  0, 0,
         0,  0, 1;

    std::array<Matrix3d, 4> Rs;
    std::array<Vector3d, 4> ts;
    Rs[0] = U * W * V.transpose();
    Rs[1] = U * W * V.transpose();
    Rs[2] = U * W.transpose() * V.transpose();
    Rs[3] = U * W.transpose() * V.transpose();
    ts[0] =  U.col(2);
    ts[1] = -U.col(2);
    ts[2] =  U.col(2);
    ts[3] = -U.col(2);

    int best_count = -1;
    int best_idx = 0;
    for (int k = 0; k < 4; k++) {
        int count = 0;
        for (size_t i = 0; i < q1.size(); i++) {
            Matrix<double, 4, 4> A;
            A.row(0) = q1[i](0) * RowVector4d(0,0,1,0) - RowVector4d(1,0,0,0);
            A.row(1) = q1[i](1) * RowVector4d(0,0,1,0) - RowVector4d(0,1,0,0);

            Matrix<double, 3, 4> P2;
            P2.block<3,3>(0,0) = Rs[k];
            P2.col(3) = ts[k];
            A.row(2) = q2[i](0) * P2.row(2) - P2.row(0);
            A.row(3) = q2[i](1) * P2.row(2) - P2.row(1);

            if (!A.allFinite()) continue;  // guard against NaN from degenerate E
            JacobiSVD<Matrix<double, 4, 4>> svd4(A, ComputeFullV);
            Vector4d X_h = svd4.matrixV().col(3);
            if (std::abs(X_h(3)) < 1e-12) continue;
            Vector3d X = X_h.head<3>() / X_h(3);

            if (X(2) > 0 && (Rs[k] * X + ts[k])(2) > 0)
                count++;
        }
        if (count > best_count) {
            best_count = count;
            best_idx = k;
        }
    }

    return {Rs[best_idx], ts[best_idx]};
}

// Fast variant: cheirality via closed-form two-view depths instead of a
// 4x4 SVD triangulation per point/candidate. For each candidate (R, t) the
// depths solve  d1 * R q1 + t = d2 * q2  in least squares (2x2 normal
// equations); the winning candidate maximizes the positive-depth count —
// the same geometric criterion as the SVD-based version.
inline Pose decompose_essential_fast(const Matrix3d& E_in,
                                     const std::vector<Vector3d>& q1,
                                     const std::vector<Vector3d>& q2)
{
    Matrix3d E = E_in.transpose();

    JacobiSVD<Matrix3d> svd(E, ComputeFullU | ComputeFullV);
    Matrix3d U = svd.matrixU();
    Matrix3d V = svd.matrixV();

    if (U.determinant() < 0) U.col(2) *= -1;
    if (V.determinant() < 0) V.col(2) *= -1;

    Matrix3d W;
    W << 0, -1, 0,
         1,  0, 0,
         0,  0, 1;

    const Matrix3d R1 = U * W * V.transpose();
    const Matrix3d R2 = U * W.transpose() * V.transpose();
    const Vector3d u2 = U.col(2);

    std::array<Matrix3d, 4> Rs = {R1, R1, R2, R2};
    std::array<Vector3d, 4> ts = {u2, -u2, u2, -u2};

    int best_count = -1;
    int best_idx = 0;
    for (int k = 0; k < 4; k++) {
        int count = 0;
        for (size_t i = 0; i < q1.size(); i++) {
            const Vector3d a = Rs[k] * q1[i];      // d1*a + t = d2*b
            const Vector3d& b = q2[i];
            const double aa = a.squaredNorm();
            const double bb = b.squaredNorm();
            const double ab = a.dot(b);
            const double det = aa * bb - ab * ab;
            if (det < 1e-14) continue;             // parallel rays
            const double at = a.dot(ts[k]);
            const double bt = b.dot(ts[k]);
            const double d1 = (ab * bt - bb * at) / det;
            const double d2 = (aa * bt - ab * at) / det;
            if (d1 > 0 && d2 > 0) count++;
        }
        if (count > best_count) {
            best_count = count;
            best_idx = k;
        }
    }

    return {Rs[best_idx], ts[best_idx]};
}

// ============================================================
// Pose-only refinement from epipolar constraints (GN)
// ============================================================

inline Pose refine_pose_epipolar(
    const Matrix3d& R0_init, const Vector3d& t_init,
    const std::vector<AffineCorrespondence>& acs,
    int max_iter = 20)
{
    int N = static_cast<int>(acs.size());
    Matrix3d R0 = R0_init;
    Vector3d t = t_init;

    for (int iter = 0; iter < max_iter; iter++) {
        Matrix3d E0 = skew(t) * R0;
        VectorXd r(N);

        // Analytic Jacobian of r_i = q2^T [t]x R0 q1:
        //   d/dw_k (R0 -> R0 exp([w]x)):  J = q1 x (R0^T (q2 x t))
        //   d/dt_k (t renormalized):      J = (I - t t^T) (R0 q1 x q2)
        Eigen::Matrix<double, Eigen::Dynamic, 6> J(N, 6);
        const Matrix3d Pt = Matrix3d::Identity() - t * t.transpose();
        for (int i = 0; i < N; i++) {
            const Vector3d& q1 = acs[i].q1;
            const Vector3d& q2 = acs[i].q2;
            r(i) = q2.dot(E0 * q1);
            const Vector3d w = R0.transpose() * (q2.cross(t));
            J.row(i).head<3>() = q1.cross(w).transpose();
            J.row(i).tail<3>() = (Pt * ((R0 * q1).cross(q2))).transpose();
        }

        Eigen::Matrix<double, 6, 6> JTJ = J.transpose() * J;
        Eigen::Matrix<double, 6, 1> JTr = J.transpose() * r;
        double lam = 0.001;
        for (int j = 0; j < 6; j++) JTJ(j, j) *= (1.0 + lam);
        Eigen::Matrix<double, 6, 1> x = JTJ.ldlt().solve(-JTr);

        R0 = R0 * axis_angle_to_rot(x.segment<3>(0));
        t = (t + x.segment<3>(3)).normalized();

        if (x.norm() < 1e-10) break;
    }
    return {R0, t};
}

// Overload for point vectors (used in PC solver path)
inline Pose refine_pose_epipolar(
    const Matrix3d& R0_init, const Vector3d& t_init,
    const std::vector<Vector3d>& q1s, const std::vector<Vector3d>& q2s,
    int max_iter = 20)
{
    int N = static_cast<int>(q1s.size());
    Matrix3d R0 = R0_init;
    Vector3d t = t_init;

    for (int iter = 0; iter < max_iter; iter++) {
        Matrix3d E0 = skew(t) * R0;
        VectorXd r(N);
        for (int i = 0; i < N; i++)
            r(i) = q2s[i].dot(E0 * q1s[i]);

        MatrixXd J(N, 6);
        double eps = 1e-7;
        for (int k = 0; k < 6; k++) {
            Matrix3d Rp = R0;
            Vector3d tp = t;
            if (k < 3) {
                Vector3d dr = Vector3d::Zero(); dr(k) = eps;
                Rp = R0 * axis_angle_to_rot(dr);
            } else {
                tp(k - 3) += eps;
                tp.normalize();
            }
            Matrix3d Ep = skew(tp) * Rp;
            VectorXd rp(N);
            for (int i = 0; i < N; i++)
                rp(i) = q2s[i].dot(Ep * q1s[i]);
            J.col(k) = (rp - r) / eps;
        }

        MatrixXd JTJ = J.transpose() * J;
        VectorXd JTr = J.transpose() * r;
        double lam = 0.001;
        for (int j = 0; j < 6; j++) JTJ(j, j) *= (1.0 + lam);
        VectorXd x = JTJ.ldlt().solve(-JTr);

        R0 = R0 * axis_angle_to_rot(x.segment<3>(0));
        t = (t + x.segment<3>(3)).normalized();

        if (x.norm() < 1e-10) break;
    }
    return {R0, t};
}

// ============================================================
// Linearized RS estimation from affine correspondences (known pose)
// ============================================================

inline RSParams estimate_rs_params_ac(
    const Matrix3d& R0, const Vector3d& t,
    const std::vector<AffineCorrespondence>& acs,
    double fy_over_h)
{
    int N = static_cast<int>(acs.size());
    assert(N >= 4);

    MatrixXd A(3 * N, 12);
    VectorXd b(3 * N);

    Matrix3d E0 = skew(t) * R0;

    // Point-independent base matrices; per-point entries only rescale by tau:
    //   dE[k]   = -tau1 * base_w1[k],  dE[3+k] = -tau1 * base_v1[k],
    //   dE[6+k] =  tau2 * base_w2[k],  dE[9+k] =  tau2 * base_v2[k],
    //   dEdtau1_dk[k]   = -base_w1[k], dEdtau1_dk[3+k] = -base_v1[k],
    //   dEdtau2_dk[6+k] =  base_w2[k], dEdtau2_dk[9+k] =  base_v2[k].
    const Matrix3d tx = skew(t);
    const Matrix3d txR0 = tx * R0;
    std::array<Matrix3d, 12> base;
    for (int k = 0; k < 3; k++) {
        Vector3d ek = Vector3d::Unit(k);
        base[k]     = txR0 * skew(ek);            // w1
        base[3+k]   = skew(R0.col(k)) * R0;       // v1  (R0*ek = R0.col(k))
        base[6+k]   = tx * skew(ek) * R0;         // w2
        base[9+k]   = skew(ek) * R0;              // v2
    }

    for (int i = 0; i < N; i++) {
        const auto& ac = acs[i];
        const double t1 = ac.tau1, t2 = ac.tau2;
        const Vector3d& q1 = ac.q1;
        const Vector3d& q2 = ac.q2;
        const Vector3d dq2u = ac.dq2_dq1.col(0);
        const Vector3d dq2v = ac.dq2_dq1.col(1);

        // Bilinear forms for a matrix M:
        //   p0 = q2^T M q1, pu = dq2u^T M q1 + (M^T q2)(0),
        //   pv = dq2v^T M q1 + (M^T q2)(1)
        const Vector3d E0q1 = E0 * q1;
        const Vector3d E0Tq2 = E0.transpose() * q2;
        b(3*i)   = -q2.dot(E0q1);
        b(3*i+1) = -(dq2u.dot(E0q1) + E0Tq2(0));
        b(3*i+2) = -(dq2v.dot(E0q1) + E0Tq2(1));

        for (int k = 0; k < 12; k++) {
            const Matrix3d& M = base[k];
            const Vector3d Mq1 = M * q1;
            const double p0 = q2.dot(Mq1);
            const double pu = dq2u.dot(Mq1) + M.col(0).dot(q2);
            const double pv = dq2v.dot(Mq1) + M.col(1).dot(q2);

            // sign * tau scaling of the main term
            const double sc = (k < 6) ? -t1 : t2;
            A(3*i,   k) = sc * p0;
            double au = sc * pu;
            double av = sc * pv;
            if (k < 6) {
                // dEdtau1 correction on the v-row only
                av += -p0 * fy_over_h;
            } else {
                // dEdtau2 corrections on both affine rows
                au += p0 * fy_over_h * dq2u(1);
                av += p0 * fy_over_h * dq2v(1);
            }
            A(3*i+1, k) = au;
            A(3*i+2, k) = av;
        }
    }

    // Truncated LS via the normal equations: eigenvectors of A^T A are the
    // right singular vectors of A and eigenvalues are the squared singular
    // values, so the same sigma > sigma_max*1e-3 truncation applies (the
    // 1e-3 relative cut is far above the sqrt of double precision, making
    // the squared conditioning harmless).
    Eigen::Matrix<double, 12, 12> AtA = A.transpose() * A;
    Eigen::Matrix<double, 12, 1> Atb = A.transpose() * b;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 12, 12>> eig(AtA);
    const auto& evals = eig.eigenvalues();   // ascending
    const auto& evecs = eig.eigenvectors();
    const double lam_thresh = evals(11) * 1e-6;  // (sigma_max * 1e-3)^2
    Eigen::Matrix<double, 12, 1> x = Eigen::Matrix<double, 12, 1>::Zero();
    for (int k = 0; k < 12; k++) {
        if (evals(k) > lam_thresh)
            x += (evecs.col(k).dot(Atb) / evals(k)) * evecs.col(k);
    }

    return RSParams(x.segment<3>(0), x.segment<3>(3),
                    x.segment<3>(6), x.segment<3>(9));
}

// ============================================================
// Joint linearized pose + RS estimation from affine correspondences
// Solves for [delta_r(3), delta_t(3), w1(3), v1(3), w2(3), v2(3)]
// simultaneously, correcting both the 5pt pose bias and RS params.
// 3N equations in 18 unknowns (minimum-norm SVD for N=5).
// ============================================================

struct PoseRSResult {
    Matrix3d R;
    Vector3d t;
    RSParams rs;
    bool valid;
};

inline PoseRSResult estimate_pose_rs_joint_ac(
    const Matrix3d& R_init, const Vector3d& t_init,
    const std::vector<AffineCorrespondence>& acs,
    double fy_over_h,
    int max_iter = 5)
{
    const int N = static_cast<int>(acs.size());
    assert(N >= 5);
    // 15 unknowns: [delta_r(3), w1(3), v1(3), w2(3), v2(3)]
    // Translation fixed from 5pt (well-determined direction).
    // Rotation corrected jointly with RS params.
    constexpr int NP = 15; // 3 pose_rot + 12 RS
    const Matrix3d I3 = Matrix3d::Identity();
    const Vector3d eu(1,0,0), ev(0,1,0);

    Matrix3d R = R_init;
    const Vector3d& t = t_init;
    RSParams rs; // initialized to zero

    // First: linearized RS with fixed pose to seed the RS params
    rs = estimate_rs_params_ac(R, t, acs, fy_over_h);

    auto compute_c1c2 = [&](const Matrix3d& Rc, const Vector3d& tc,
                             const RSParams& rsc, const AffineCorrespondence& ac)
        -> std::pair<double,double>
    {
        double t1 = ac.tau1, t2 = ac.tau2;
        Matrix3d Aa = (I3 + t2*skew(rsc.omega2)) * Rc;
        Matrix3d Bb = I3 - t1*skew(rsc.omega1);
        Matrix3d Rt = Aa * Bb;
        Vector3d tt = (tc + t2*rsc.v2) - t1*Rt*rsc.v1;

        Matrix3d dRt1 = Aa * (-skew(rsc.omega1));
        Vector3d dtt1 = -Rt*rsc.v1 - t1*dRt1*rsc.v1;
        Matrix3d dE1 = skew(dtt1) * Rt + skew(tt) * dRt1;

        Matrix3d dRt2 = skew(rsc.omega2) * Rc * Bb;
        Vector3d dtt2 = rsc.v2 - t1*dRt2*rsc.v1;
        Matrix3d dE2 = skew(dtt2) * Rt + skew(tt) * dRt2;

        return {ac.q2.dot(dE1 * ac.q1) * fy_over_h,
                ac.q2.dot(dE2 * ac.q1) * fy_over_h};
    };

    for (int iter = 0; iter < max_iter; iter++) {
        const int M = 3 * N;
        VectorXd residuals(M);
        MatrixXd Jac(M, NP);

        for (int i = 0; i < N; i++) {
            const auto& ac = acs[i];
            double t1 = ac.tau1, t2 = ac.tau2;

            Matrix3d Aa = (I3 + t2*skew(rs.omega2)) * R;
            Matrix3d B = I3 - t1*skew(rs.omega1);
            Matrix3d Rt = Aa * B;
            Vector3d tt = (t + t2*rs.v2) - t1*Rt*rs.v1;
            Matrix3d Ei = skew(tt) * Rt;

            auto [c1, c2] = compute_c1c2(R, t, rs, ac);
            Vector3d du = ac.dq2_dq1.col(0), dv = ac.dq2_dq1.col(1);

            residuals(3*i)     = ac.q2.dot(Ei * ac.q1);
            residuals(3*i + 1) = du.dot(Ei * ac.q1) + ac.q2.dot(Ei * eu) + c2 * du(1);
            residuals(3*i + 2) = dv.dot(Ei * ac.q1) + ac.q2.dot(Ei * ev) + c1 + c2 * dv(1);

            Matrix3d M0 = ac.q2 * ac.q1.transpose();
            Matrix3d M1 = du * ac.q1.transpose() + ac.q2 * eu.transpose();
            Matrix3d M2 = dv * ac.q1.transpose() + ac.q2 * ev.transpose();

            auto frob = [](const Matrix3d& X, const Matrix3d& Y) -> double {
                return (X.array() * Y.array()).sum();
            };

            const double eps = 1e-7;

            // --- Rotation columns (0-2): delta_r ---
            // R -> R * (I + [dr]x)
            // Rt_new = (I + t2*[w2]x) * R * (I + [dr]x) * (I - t1*[w1]x)
            // dRt/d(dr_k) = Aa * skew(ek) * B
            // dtt/d(dr_k) = -t1 * dRt/d(dr_k) * v1
            for (int k = 0; k < 3; k++) {
                Vector3d ek = Vector3d::Unit(k);
                Matrix3d dRt_dr = Aa * skew(ek) * B;
                Vector3d dtt_dr = -t1 * dRt_dr * rs.v1;
                Matrix3d dE_dr = skew(dtt_dr) * Rt + skew(tt) * dRt_dr;

                Jac(3*i,     k) = frob(dE_dr, M0);
                Jac(3*i + 1, k) = frob(dE_dr, M1);
                Jac(3*i + 2, k) = frob(dE_dr, M2);

                // c1/c2 FD correction for rotation
                Matrix3d R_p = R * axis_angle_to_rot(eps * ek);
                auto [c1_p, c2_p] = compute_c1c2(R_p, t, rs, ac);
                double dc1 = (c1_p - c1) / eps;
                double dc2 = (c2_p - c2) / eps;
                Jac(3*i + 1, k) += dc2 * du(1);
                Jac(3*i + 2, k) += dc1 + dc2 * dv(1);
            }

            // --- RS columns (3-14): w1, v1, w2, v2 ---
            for (int p = 0; p < 12; p++) {
                Matrix3d dRt_dp;
                Vector3d dtt_dp;

                if (p < 3) {
                    Vector3d ek = Vector3d::Unit(p);
                    dRt_dp = -t1 * Aa * skew(ek);
                    dtt_dp = -t1 * dRt_dp * rs.v1;
                } else if (p < 6) {
                    Vector3d ek = Vector3d::Unit(p - 3);
                    dRt_dp.setZero();
                    dtt_dp = -t1 * Rt * ek;
                } else if (p < 9) {
                    Vector3d ek = Vector3d::Unit(p - 6);
                    dRt_dp = t2 * skew(ek) * R * B;
                    dtt_dp = -t1 * dRt_dp * rs.v1;
                } else {
                    Vector3d ek = Vector3d::Unit(p - 9);
                    dRt_dp.setZero();
                    dtt_dp = t2 * ek;
                }

                Matrix3d dE_dp = skew(dtt_dp) * Rt + skew(tt) * dRt_dp;

                Jac(3*i,     3+p) = frob(dE_dp, M0);
                Jac(3*i + 1, 3+p) = frob(dE_dp, M1);
                Jac(3*i + 2, 3+p) = frob(dE_dp, M2);

                RSParams rs_p = rs;
                if (p < 3) rs_p.omega1(p) += eps;
                else if (p < 6) rs_p.v1(p - 3) += eps;
                else if (p < 9) rs_p.omega2(p - 6) += eps;
                else rs_p.v2(p - 9) += eps;

                auto [c1_p, c2_p] = compute_c1c2(R, t, rs_p, ac);
                double dc1 = (c1_p - c1) / eps;
                double dc2 = (c2_p - c2) / eps;

                Jac(3*i + 1, 3+p) += dc2 * du(1);
                Jac(3*i + 2, 3+p) += dc1 + dc2 * dv(1);
            }
        }

        VectorXd delta = Jac.colPivHouseholderQr().solve(-residuals);
        if (delta.norm() < 1e-12) break;

        // Apply rotation update
        R = R * axis_angle_to_rot(delta.segment<3>(0));

        // Apply RS update
        rs.omega1 += delta.segment<3>(3);
        rs.v1     += delta.segment<3>(6);
        rs.omega2 += delta.segment<3>(9);
        rs.v2     += delta.segment<3>(12);
    }

    return { R, t, rs, true };
}

// ============================================================
// Newton-refined RS from affine correspondences (known pose)
// ============================================================

inline RSParams estimate_rs_params_exact_ac(
    const Matrix3d& R, const Vector3d& t,
    const std::vector<AffineCorrespondence>& acs,
    double fy_over_h,
    int newton_steps = 2,
    const RSParams* rs_init = nullptr)
{
    constexpr int NP = 12;
    const int N = static_cast<int>(acs.size());
    const Matrix3d I3 = Matrix3d::Identity();
    const Vector3d eu(1,0,0), ev(0,1,0);

    RSParams rs;
    if (rs_init) {
        rs = *rs_init;
    } else {
        rs = estimate_rs_params_ac(R, t, acs, fy_over_h);
    }

    auto compute_c1c2 = [&](const RSParams& rsc, const AffineCorrespondence& ac) -> std::pair<double,double> {
        double t1 = ac.tau1, t2 = ac.tau2;
        Matrix3d Aa = (I3 + t2*skew(rsc.omega2)) * R;
        Matrix3d Bb = I3 - t1*skew(rsc.omega1);
        Matrix3d Rt = Aa * Bb;
        Vector3d tt = (t + t2*rsc.v2) - t1*Rt*rsc.v1;

        Matrix3d dRt1 = Aa * (-skew(rsc.omega1));
        Vector3d dtt1 = -Rt*rsc.v1 - t1*dRt1*rsc.v1;
        Matrix3d dE1 = skew(dtt1) * Rt + skew(tt) * dRt1;

        Matrix3d dRt2 = skew(rsc.omega2) * R * Bb;
        Vector3d dtt2 = rsc.v2 - t1*dRt2*rsc.v1;
        Matrix3d dE2 = skew(dtt2) * Rt + skew(tt) * dRt2;

        return {ac.q2.dot(dE1 * ac.q1) * fy_over_h,
                ac.q2.dot(dE2 * ac.q1) * fy_over_h};
    };

    for (int step = 0; step < newton_steps; step++) {
        const int M = 3 * N;
        VectorXd residuals(M);
        MatrixXd Jac(M, NP);

        for (int i = 0; i < N; i++) {
            const auto& ac = acs[i];
            double t1 = ac.tau1, t2 = ac.tau2;

            Matrix3d Aa = (I3 + t2*skew(rs.omega2)) * R;
            Matrix3d B = I3 - t1*skew(rs.omega1);
            Matrix3d Rt = Aa * B;
            Vector3d tt = (t + t2*rs.v2) - t1*Rt*rs.v1;
            Matrix3d Ei = skew(tt) * Rt;

            auto [c1, c2] = compute_c1c2(rs, ac);
            Vector3d du = ac.dq2_dq1.col(0), dv = ac.dq2_dq1.col(1);

            residuals(3*i)     = ac.q2.dot(Ei * ac.q1);
            residuals(3*i + 1) = du.dot(Ei * ac.q1) + ac.q2.dot(Ei * eu) + c2 * du(1);
            residuals(3*i + 2) = dv.dot(Ei * ac.q1) + ac.q2.dot(Ei * ev) + c1 + c2 * dv(1);

            Matrix3d M0 = ac.q2 * ac.q1.transpose();
            Matrix3d M1 = du * ac.q1.transpose() + ac.q2 * eu.transpose();
            Matrix3d M2 = dv * ac.q1.transpose() + ac.q2 * ev.transpose();

            auto frob = [](const Matrix3d& X, const Matrix3d& Y) -> double {
                return (X.array() * Y.array()).sum();
            };

            const double eps = 1e-7;
            for (int p = 0; p < NP; p++) {
                Matrix3d dRt_dp;
                Vector3d dtt_dp;

                if (p < 3) {
                    Vector3d ek = Vector3d::Unit(p);
                    dRt_dp = -t1 * Aa * skew(ek);
                    dtt_dp = -t1 * dRt_dp * rs.v1;
                } else if (p < 6) {
                    Vector3d ek = Vector3d::Unit(p - 3);
                    dRt_dp.setZero();
                    dtt_dp = -t1 * Rt * ek;
                } else if (p < 9) {
                    Vector3d ek = Vector3d::Unit(p - 6);
                    dRt_dp = t2 * skew(ek) * R * B;
                    dtt_dp = -t1 * dRt_dp * rs.v1;
                } else {
                    Vector3d ek = Vector3d::Unit(p - 9);
                    dRt_dp.setZero();
                    dtt_dp = t2 * ek;
                }

                Matrix3d dE_dp = skew(dtt_dp) * Rt + skew(tt) * dRt_dp;

                Jac(3*i,     p) = frob(dE_dp, M0);
                Jac(3*i + 1, p) = frob(dE_dp, M1);
                Jac(3*i + 2, p) = frob(dE_dp, M2);

                RSParams rs_p = rs;
                if (p < 3) rs_p.omega1(p) += eps;
                else if (p < 6) rs_p.v1(p - 3) += eps;
                else if (p < 9) rs_p.omega2(p - 6) += eps;
                else rs_p.v2(p - 9) += eps;

                auto [c1_p, c2_p] = compute_c1c2(rs_p, ac);
                double dc1 = (c1_p - c1) / eps;
                double dc2 = (c2_p - c2) / eps;

                Jac(3*i + 1, p) += dc2 * du(1);
                Jac(3*i + 2, p) += dc1 + dc2 * dv(1);
            }
        }

        VectorXd delta = Jac.colPivHouseholderQr().solve(-residuals);
        if (delta.norm() < 1e-12) break;

        rs.omega1 += delta.segment<3>(0);
        rs.v1     += delta.segment<3>(3);
        rs.omega2 += delta.segment<3>(6);
        rs.v2     += delta.segment<3>(9);
    }

    return rs;
}

// ============================================================
// Joint pose + RS refinement via LM (PC-based, 18 params)
// ============================================================

inline JointPoseRS estimate_joint_pose_rs(
    const Matrix3d& R0_init, const Vector3d& t_init,
    const std::vector<Vector3d>& q1, const std::vector<Vector3d>& q2,
    const std::vector<double>& tau1, const std::vector<double>& tau2,
    int refine_iters = 10,
    const RSParams* rs_init = nullptr)
{
    int N = static_cast<int>(q1.size());
    Matrix3d R0 = R0_init;
    Vector3d t = t_init;
    RSParams rs = rs_init ? *rs_init : RSParams();

    for (int ref = 0; ref < refine_iters; ref++) {
        MatrixXd A(N, 18);
        VectorXd b(N);

        for (int i = 0; i < N; i++) {
            Matrix3d R_tilde = (Matrix3d::Identity() + tau2[i] * skew(rs.omega2)) * R0 *
                               (Matrix3d::Identity() - tau1[i] * skew(rs.omega1));
            Vector3d t_tilde = (t + tau2[i] * rs.v2) - tau1[i] * R_tilde * rs.v1;
            Matrix3d E_i = skew(t_tilde) * R_tilde;
            b(i) = -q2[i].dot(E_i * q1[i]);

            for (int k = 0; k < 3; k++) {
                Vector3d ek = Vector3d::Unit(k);

                Matrix3d dRt_dr = (Matrix3d::Identity() + tau2[i] * skew(rs.omega2)) * R0 *
                                  skew(ek) * (Matrix3d::Identity() - tau1[i] * skew(rs.omega1));
                Vector3d dtt_dr = -tau1[i] * dRt_dr * rs.v1;
                A(i, k) = q2[i].dot((skew(dtt_dr) * R_tilde + skew(t_tilde) * dRt_dr) * q1[i]);

                A(i, 3 + k) = q2[i].dot(skew(ek) * R_tilde * q1[i]);

                Matrix3d dRt_w1 = -(Matrix3d::Identity() + tau2[i] * skew(rs.omega2)) * R0 *
                                   tau1[i] * skew(ek);
                Vector3d dtt_w1 = -tau1[i] * dRt_w1 * rs.v1;
                A(i, 6 + k) = q2[i].dot((skew(dtt_w1) * R_tilde + skew(t_tilde) * dRt_w1) * q1[i]);

                A(i, 9 + k) = q2[i].dot(skew(-tau1[i] * R_tilde * ek) * R_tilde * q1[i]);

                Matrix3d dRt_w2 = tau2[i] * skew(ek) * R0 *
                                  (Matrix3d::Identity() - tau1[i] * skew(rs.omega1));
                Vector3d dtt_w2 = -tau1[i] * dRt_w2 * rs.v1;
                A(i, 12 + k) = q2[i].dot((skew(dtt_w2) * R_tilde + skew(t_tilde) * dRt_w2) * q1[i]);

                A(i, 15 + k) = q2[i].dot(skew(tau2[i] * ek) * R_tilde * q1[i]);
            }
        }

        MatrixXd ATA = A.transpose() * A;
        VectorXd ATb = A.transpose() * b;
        double lam = 0.05 * ATA.diagonal().mean();
        for (int j = 0; j < 6; j++)    ATA(j, j) += lam;
        for (int j = 9; j < 12; j++)   ATA(j, j) += lam;
        for (int j = 15; j < 18; j++)  ATA(j, j) += lam;
        VectorXd x = ATA.ldlt().solve(ATb);

        R0 = R0 * axis_angle_to_rot(x.segment<3>(0));
        t = (t + x.segment<3>(3)).normalized();

        rs.omega1 += x.segment<3>(6);
        rs.v1     += x.segment<3>(9);
        rs.omega2 += x.segment<3>(12);
        rs.v2     += x.segment<3>(15);

        if (x.norm() < 1e-10) break;
    }

    return {R0, t, rs};
}

// ============================================================
// Joint pose + RS refinement via LM (AC-based, 17 params)
// ============================================================

inline JointPoseRS estimate_joint_pose_rs_ac(
    const Matrix3d& R_init, const Vector3d& t_init,
    const std::vector<AffineCorrespondence>& acs,
    double fy_over_h,
    int max_iter = 100,
    bool verbose = false,
    const RSParams* rs_init = nullptr,
    double ac_weight = 1.0)
{
    constexpr int NP = 17;
    const int N = static_cast<int>(acs.size());
    const Matrix3d I3 = Matrix3d::Identity();
    const Vector3d eu(1,0,0), ev(0,1,0);
    const double w_ac = ac_weight;

    Matrix3d R = R_init;
    Vector3d t = t_init.normalized();
    RSParams rs = rs_init ? *rs_init : RSParams();

    auto compute_cost = [&](const Matrix3d& Rc, const Vector3d& tc, const RSParams& rsc) -> double {
        double cost = 0;
        for (int i = 0; i < N; i++) {
            const auto& ac = acs[i];
            double t1 = ac.tau1, t2 = ac.tau2;
            Matrix3d Rt = (I3 + t2*skew(rsc.omega2)) * Rc * (I3 - t1*skew(rsc.omega1));
            Vector3d tt = (tc + t2*rsc.v2) - t1*Rt*rsc.v1;
            Matrix3d Ei = skew(tt) * Rt;

            Matrix3d dRt1 = (I3 + t2*skew(rsc.omega2)) * Rc * (-skew(rsc.omega1));
            Vector3d dtt1 = -Rt*rsc.v1 - t1*dRt1*rsc.v1;
            Matrix3d dE1 = skew(dtt1) * Rt + skew(tt) * dRt1;

            Matrix3d dRt2 = skew(rsc.omega2) * Rc * (I3 - t1*skew(rsc.omega1));
            Vector3d dtt2 = rsc.v2 - t1*dRt2*rsc.v1;
            Matrix3d dE2 = skew(dtt2) * Rt + skew(tt) * dRt2;

            double c1 = ac.q2.dot(dE1 * ac.q1) * fy_over_h;
            double c2 = ac.q2.dot(dE2 * ac.q1) * fy_over_h;
            Vector3d du = ac.dq2_dq1.col(0), dv = ac.dq2_dq1.col(1);

            double r0 = ac.q2.dot(Ei * ac.q1);
            double r1 = (du.dot(Ei * ac.q1) + ac.q2.dot(Ei * eu) + c2 * du(1)) * w_ac;
            double r2 = (dv.dot(Ei * ac.q1) + ac.q2.dot(Ei * ev) + c1 + c2 * dv(1)) * w_ac;
            cost += r0*r0 + r1*r1 + r2*r2;
        }
        return cost;
    };

    auto compute_c1c2 = [&](const Matrix3d& Rc, const Vector3d& tc, const RSParams& rsc,
                             const AffineCorrespondence& ac) -> std::pair<double,double> {
        double t1 = ac.tau1, t2 = ac.tau2;
        Matrix3d Aa = (I3 + t2*skew(rsc.omega2)) * Rc;
        Matrix3d B = I3 - t1*skew(rsc.omega1);
        Matrix3d Rt = Aa * B;
        Vector3d tt = (tc + t2*rsc.v2) - t1*Rt*rsc.v1;

        Matrix3d dRt1 = Aa * (-skew(rsc.omega1));
        Vector3d dtt1 = -Rt*rsc.v1 - t1*dRt1*rsc.v1;
        Matrix3d dE1_ = skew(dtt1) * Rt + skew(tt) * dRt1;

        Matrix3d dRt2 = skew(rsc.omega2) * Rc * B;
        Vector3d dtt2 = rsc.v2 - t1*dRt2*rsc.v1;
        Matrix3d dE2_ = skew(dtt2) * Rt + skew(tt) * dRt2;

        return {ac.q2.dot(dE1_ * ac.q1) * fy_over_h,
                ac.q2.dot(dE2_ * ac.q1) * fy_over_h};
    };

    double cost = compute_cost(R, t, rs);
    double lambda = 1e-3;
    constexpr double LAM_MIN = 1e-10, LAM_MAX = 1e10;
    bool recompute = true;

    Matrix<double, NP, NP> JtJ;
    Matrix<double, NP, 1> Jtr;
    Matrix<double, 3, 2> tb;

    for (int iter = 0; iter < max_iter; iter++) {
        if (recompute) {
            int mi = 0;
            if (std::abs(t(1)) < std::abs(t(mi))) mi = 1;
            if (std::abs(t(2)) < std::abs(t(mi))) mi = 2;
            tb.col(0) = t.cross(Vector3d::Unit(mi)).normalized();
            tb.col(1) = tb.col(0).cross(t).normalized();

            JtJ.setZero();
            Jtr.setZero();

            for (int i = 0; i < N; i++) {
                const auto& ac = acs[i];
                double t1 = ac.tau1, t2 = ac.tau2;

                Matrix3d Aa = (I3 + t2*skew(rs.omega2)) * R;
                Matrix3d B = I3 - t1*skew(rs.omega1);
                Matrix3d Rt = Aa * B;
                Vector3d tt = (t + t2*rs.v2) - t1*Rt*rs.v1;
                Matrix3d Ei = skew(tt) * Rt;

                Matrix3d dRt1 = Aa * (-skew(rs.omega1));
                Vector3d dtt1 = -Rt*rs.v1 - t1*dRt1*rs.v1;
                Matrix3d dE1_ = skew(dtt1) * Rt + skew(tt) * dRt1;

                Matrix3d dRt2 = skew(rs.omega2) * R * B;
                Vector3d dtt2 = rs.v2 - t1*dRt2*rs.v1;
                Matrix3d dE2_ = skew(dtt2) * Rt + skew(tt) * dRt2;

                double c1 = ac.q2.dot(dE1_ * ac.q1) * fy_over_h;
                double c2 = ac.q2.dot(dE2_ * ac.q1) * fy_over_h;

                Vector3d du = ac.dq2_dq1.col(0), dv = ac.dq2_dq1.col(1);

                double r0 = ac.q2.dot(Ei * ac.q1);
                double r1 = (du.dot(Ei * ac.q1) + ac.q2.dot(Ei * eu) + c2 * du(1)) * w_ac;
                double r2 = (dv.dot(Ei * ac.q1) + ac.q2.dot(Ei * ev) + c1 + c2 * dv(1)) * w_ac;

                Matrix3d M0 = ac.q2 * ac.q1.transpose();
                Matrix3d M1 = du * ac.q1.transpose() + ac.q2 * eu.transpose();
                Matrix3d M2 = dv * ac.q1.transpose() + ac.q2 * ev.transpose();

                auto frob = [](const Matrix3d& X, const Matrix3d& Y) -> double {
                    return (X.array() * Y.array()).sum();
                };

                Matrix<double, 3, NP> Ji;
                const double eps = 1e-7;

                for (int p = 0; p < NP; p++) {
                    Matrix3d dRt_dp, dE_dp;
                    Vector3d dtt_dp;

                    if (p < 3) {
                        Vector3d ek = Vector3d::Unit(p);
                        dRt_dp = Aa * skew(ek) * B;
                        dtt_dp = -t1 * dRt_dp * rs.v1;
                    } else if (p < 5) {
                        dRt_dp.setZero();
                        dtt_dp = tb.col(p - 3);
                    } else if (p < 8) {
                        Vector3d ek = Vector3d::Unit(p - 5);
                        dRt_dp = -t1 * Aa * skew(ek);
                        dtt_dp = -t1 * dRt_dp * rs.v1;
                    } else if (p < 11) {
                        Vector3d ek = Vector3d::Unit(p - 8);
                        dRt_dp.setZero();
                        dtt_dp = -t1 * Rt * ek;
                    } else if (p < 14) {
                        Vector3d ek = Vector3d::Unit(p - 11);
                        dRt_dp = t2 * skew(ek) * R * B;
                        dtt_dp = -t1 * dRt_dp * rs.v1;
                    } else {
                        Vector3d ek = Vector3d::Unit(p - 14);
                        dRt_dp.setZero();
                        dtt_dp = t2 * ek;
                    }

                    dE_dp = skew(dtt_dp) * Rt + skew(tt) * dRt_dp;

                    Ji(0, p) = frob(dE_dp, M0);
                    Ji(1, p) = frob(dE_dp, M1);
                    Ji(2, p) = frob(dE_dp, M2);

                    Matrix3d R_p = R;
                    Vector3d t_p = t;
                    RSParams rs_p = rs;

                    if (p < 3) {
                        Vector3d dw = Vector3d::Zero(); dw(p) = eps;
                        R_p = R * axis_angle_to_rot(dw);
                    } else if (p < 5) {
                        t_p = t + tb.col(p - 3) * eps;
                    } else if (p < 8) {
                        rs_p.omega1(p - 5) += eps;
                    } else if (p < 11) {
                        rs_p.v1(p - 8) += eps;
                    } else if (p < 14) {
                        rs_p.omega2(p - 11) += eps;
                    } else {
                        rs_p.v2(p - 14) += eps;
                    }

                    auto [c1_p, c2_p] = compute_c1c2(R_p, t_p, rs_p, ac);
                    double dc1 = (c1_p - c1) / eps;
                    double dc2 = (c2_p - c2) / eps;

                    Ji(1, p) += dc2 * du(1);
                    Ji(2, p) += dc1 + dc2 * dv(1);

                    Ji(1, p) *= w_ac;
                    Ji(2, p) *= w_ac;
                }

                Vector3d ri(r0, r1, r2);
                JtJ.noalias() += Ji.transpose() * Ji;
                Jtr.noalias() += Ji.transpose() * ri;
            }
        }

        double grad_norm = Jtr.norm();
        if (grad_norm < 1e-10) break;

        Matrix<double, NP, NP> H = JtJ;
        for (int j = 0; j < NP; j++) H(j, j) += lambda;
        double lam_v = 0.02 * JtJ.diagonal().mean();
        for (int j = 8; j < 11; j++)  H(j, j) += lam_v;
        for (int j = 14; j < 17; j++) H(j, j) += lam_v;
        Matrix<double, NP, 1> dx = H.llt().solve(-Jtr);

        if (dx.norm() < 1e-8) break;

        Matrix3d R_new = R * axis_angle_to_rot(dx.segment<3>(0));
        Vector3d t_new = (t + tb * dx.segment<2>(3)).normalized();
        RSParams rs_new = rs;
        rs_new.omega1 += dx.segment<3>(5);
        rs_new.v1     += dx.segment<3>(8);
        rs_new.omega2 += dx.segment<3>(11);
        rs_new.v2     += dx.segment<3>(14);

        double cost_new = compute_cost(R_new, t_new, rs_new);

        if (cost_new < cost) {
            R = R_new; t = t_new; rs = rs_new;
            cost = cost_new;
            lambda = std::max(LAM_MIN, lambda / 10.0);
            recompute = true;
        } else {
            lambda = std::min(LAM_MAX, lambda * 10.0);
            recompute = false;
        }
    }

    return {R, t, rs};
}

// ============================================================
// PC-based linearized RS extraction (12 unknowns)
// ============================================================
// Solves q2^T * E_tilde(omega,v) * q1 = 0 linearized around (R0, t).
// 12 unknowns: omega1(3), v1(3), omega2(3), v2(3).

inline RSParams estimate_rs_params_pc(
    const Matrix3d& R0, const Vector3d& t,
    const std::vector<Vector3d>& q1, const std::vector<Vector3d>& q2,
    const std::vector<double>& tau1, const std::vector<double>& tau2)
{
    int N = static_cast<int>(q1.size());
    if (N < 12) return RSParams();

    MatrixXd A(N, 12);
    VectorXd b(N);

    Matrix3d E0 = skew(t) * R0;
    for (int i = 0; i < N; i++) {
        b(i) = -q2[i].dot(E0 * q1[i]);
        for (int k = 0; k < 3; k++) {
            Vector3d ek = Vector3d::Unit(k);
            A(i, k)     = q2[i].dot(-tau1[i] * skew(t) * R0 * skew(ek) * q1[i]);
            A(i, 3 + k) = q2[i].dot(skew(-tau1[i] * R0 * ek) * R0 * q1[i]);
            A(i, 6 + k) = q2[i].dot(tau2[i] * skew(t) * skew(ek) * R0 * q1[i]);
            A(i, 9 + k) = q2[i].dot(skew(tau2[i] * ek) * R0 * q1[i]);
        }
    }

    VectorXd x = A.colPivHouseholderQr().solve(b);
    return RSParams(x.segment<3>(0), x.segment<3>(3),
                    x.segment<3>(6), x.segment<3>(9));
}

// ============================================================
// Constant-velocity RS extraction (6 unknowns)
// ============================================================
// Assumes omega1 = omega2 = omega, v1 = v2 = v.

inline RSParams estimate_rs_params_cv(
    const Matrix3d& R0, const Vector3d& t,
    const std::vector<Vector3d>& q1, const std::vector<Vector3d>& q2,
    const std::vector<double>& tau1, const std::vector<double>& tau2)
{
    int N = static_cast<int>(q1.size());
    if (N < 6) return RSParams();

    MatrixXd A(N, 6);
    VectorXd b(N);

    Matrix3d E0 = skew(t) * R0;
    for (int i = 0; i < N; i++) {
        b(i) = -q2[i].dot(E0 * q1[i]);
        for (int k = 0; k < 3; k++) {
            Vector3d ek = Vector3d::Unit(k);
            double dw1 = q2[i].dot(-tau1[i] * skew(t) * R0 * skew(ek) * q1[i]);
            double dw2 = q2[i].dot(tau2[i] * skew(t) * skew(ek) * R0 * q1[i]);
            A(i, k) = dw1 + dw2;
            double dv1 = q2[i].dot(skew(-tau1[i] * R0 * ek) * R0 * q1[i]);
            double dv2 = q2[i].dot(skew(tau2[i] * ek) * R0 * q1[i]);
            A(i, 3 + k) = dv1 + dv2;
        }
    }

    VectorXd x = A.colPivHouseholderQr().solve(b);
    RSParams rs;
    rs.omega1 = x.segment<3>(0);
    rs.v1     = x.segment<3>(3);
    rs.omega2 = rs.omega1;
    rs.v2     = rs.v1;
    return rs;
}

// ============================================================
// Extract point correspondences from DataMatrix
// ============================================================

inline void extract_points_from_data(
    const DataMatrix& data, const size_t* sample, size_t n,
    std::vector<Vector3d>& q1, std::vector<Vector3d>& q2,
    std::vector<double>& tau1, std::vector<double>& tau2)
{
    q1.resize(n); q2.resize(n);
    tau1.resize(n); tau2.resize(n);
    for (size_t i = 0; i < n; i++) {
        size_t idx = sample ? sample[i] : i;
        q1[i] = Vector3d(data(idx, 0), data(idx, 1), 1.0);
        q2[i] = Vector3d(data(idx, 2), data(idx, 3), 1.0);
        tau1[i] = data(idx, 4);
        tau2[i] = data(idx, 5);
    }
}

// ============================================================
// 8-point essential matrix solver
// ============================================================

inline Pose solve_8point_essential(
    const std::vector<Vector3d>& q1, const std::vector<Vector3d>& q2)
{
    int N = static_cast<int>(q1.size());
    assert(N >= 8);

    MatrixXd A(N, 9);
    for (int i = 0; i < N; i++)
        for (int a = 0; a < 3; a++)
            for (int b = 0; b < 3; b++)
                A(i, 3*a + b) = q2[i](a) * q1[i](b);

    JacobiSVD<MatrixXd> svd(A, ComputeFullV);
    VectorXd e = svd.matrixV().col(8);

    Matrix3d E;
    for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++)
            E(a,b) = e(3*a + b);

    // Project to essential matrix manifold
    JacobiSVD<Matrix3d> svd_E(E, ComputeFullU | ComputeFullV);
    Vector3d sv = svd_E.singularValues();
    double s = (sv(0) + sv(1)) / 2.0;
    E = svd_E.matrixU() * Vector3d(s, s, 0).asDiagonal() *
        svd_E.matrixV().transpose();

    return decompose_essential(E.transpose(), q1, q2);
}

// ============================================================
// Helper: extract AffineCorrespondence from DataMatrix row
// ============================================================

inline AffineCorrespondence ac_from_data_row(const DataMatrix& data, size_t idx) {
    AffineCorrespondence ac;
    ac.q1 = Vector3d(data(idx, 0), data(idx, 1), 1.0);
    ac.q2 = Vector3d(data(idx, 2), data(idx, 3), 1.0);
    ac.tau1 = data(idx, 4);
    ac.tau2 = data(idx, 5);
    ac.dq2_dq1.col(0) = Vector3d(data(idx, 6), data(idx, 7), data(idx, 8));
    ac.dq2_dq1.col(1) = Vector3d(data(idx, 9), data(idx, 10), data(idx, 11));
    return ac;
}

// ============================================================
// Helper: pack/unpack model from 8x3 DataMatrix
// ============================================================

inline void pack_model(DataMatrix& out, const Matrix3d& R, const Vector3d& t, const RSParams& rs) {
    out.resize(8, 3);
    out.block<3,3>(0,0) = R;
    out.row(3) = t.transpose();
    out.row(4) = rs.omega1.transpose();
    out.row(5) = rs.v1.transpose();
    out.row(6) = rs.omega2.transpose();
    out.row(7) = rs.v2.transpose();
}

inline void unpack_model(const DataMatrix& data, Matrix3d& R, Vector3d& t, RSParams& rs) {
    R = data.block<3,3>(0,0);
    t = data.row(3).transpose();
    rs.omega1 = data.row(4).transpose();
    rs.v1 = data.row(5).transpose();
    rs.omega2 = data.row(6).transpose();
    rs.v2 = data.row(7).transpose();
}

} // namespace rs
} // namespace estimator
} // namespace superansac
