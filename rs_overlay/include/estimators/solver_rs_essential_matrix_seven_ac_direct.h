// RS Essential Matrix 7-AC Direct Solver (Action Matrix)
//
// Algebraic solver based on minor elimination + d(s)-divided action matrices.
//
// Pipeline:
//   1. 5pt Nister on first 5 ACs → best E via remaining 2 → decompose → refine
//   2. Left null space U_null of A_ref at s_ref=0 via Householder QR
//   3. Exact closed-form G_poly (9×3, deg 2): the residual b(E) is linear in
//      E and d(s)R(s) is polynomial, so no sampling/fitting is needed
//   4. 5 fixed random linear combos of G rows → 5×3 polynomial matrix
//   5. C(5,3)=10 3×3 minors (shared 2×2 cofactors) → degree-6 polys
//   6. Divide each by d(s) = 1+|s|² via precomputed pseudo-inverse → deg 4
//   7. Macaulay matrix at d_ext=5 (40×56, precomputed index tables)
//      → 20-dim null space via rank-revealing QR of C^T
//   8. Action matrix A_s1 = V_null^T S_1 V_null (20×20, fixed-size)
//   9. Eigendecompose A_s1^T; s2, s3 from monomial-vector ratios of the
//      (left) eigenvectors mapped back through V_null
//  10. Recover (u,v) from G(s)*[1,u,v]=0 → R = cayley_to_R(s), t = [1,u,v]/norm
//
// Algebraic degree: 20 (confirmed via Macaulay H-function stabilization)
// Sample size: 7 affine correspondences
// Measured runtime (i9-10900K, single core): ~0.15ms algebraic part,
// ~0.20ms per estimateModel call incl. 5pt init and per-model RS params
// (previous implementation: ~0.73ms)
#pragma once

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <vector>
#include <array>
#include <map>
#include <tuple>
#include <cmath>
#include <algorithm>
#include <complex>
#include <random>
#include <cassert>

#include "abstract_solver.h"
#include "solver_essential_matrix_five_point_nister.h"
#include "solver_rs_essential_matrix_six_ac_eigval.h"  // reuse polynomial utilities
#include "rs_utils.h"
#include "../models/model.h"
#include "../utils/types.h"
#include "../utils/macros.h"

namespace superansac {
namespace estimator {

// ============================================================
// Direct algebraic solver utilities (7-AC, action matrix)
// ============================================================
namespace rs_direct {

using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::Vector2d;

// Reuse polynomial algebra from rs_eigval
using rs_eigval::Mono3;
using rs_eigval::GrevlexDesc;
using rs_eigval::gen_monomials;
using rs_eigval::mono_mult;
using rs_eigval::eval_mono;
using rs_eigval::eval_monos;
using rs_eigval::build_mono_index;
using rs_eigval::cayley_to_R;
using rs_eigval::R_to_cayley;
using rs_eigval::build_rs_system_Ab;
using rs_eigval::compute_fixed_null_space;

// ============================================================
// Dynamic-size G_poly coefficients (for N >= 7 ACs)
// ============================================================

struct GPolyCoeffsDyn {
    std::vector<std::array<VectorXd, 3>> coeffs; // [row][col=0,1,2] = VectorXd(10)
    std::array<MatrixXd, 3> coeff_mat;           // [col] = n_rows×10 (same data, GEMM-friendly)
    std::vector<Mono3> monoms; // 10 degree-2 monomials
    int n_rows;                // 3N - 12 (e.g., 9 for N=7)
};

// Build G(s) at a specific Cayley vector s, dynamic size
inline MatrixXd build_G_fixed_dyn(
    const Vector3d& s,
    const std::vector<rs::AffineCorrespondence>& acs,
    double fy_over_h,
    const MatrixXd& U_null)
{
    const int n_null = static_cast<int>(U_null.cols());
    Matrix3d R = cayley_to_R(s);
    MatrixXd G(n_null, 3);
    for (int k = 0; k < 3; k++) {
        Vector3d ek = Vector3d::Unit(k);
        auto [Ak, bk] = build_rs_system_Ab(R, ek, acs, fy_over_h);
        G.col(k) = U_null.transpose() * bk;
    }
    return G;
}

// ============================================================
// Closed-form G_poly coefficients
// ============================================================
// b(E) from build_rs_system_Ab is linear in E:  b = B * vecc(E), with
// vecc the column-major vectorization and B depending only on the ACs.
// With t = e_k and the Cayley parameterization, the polynomial matrix is
//   G_poly(s).col(k) = U_null^T B vecc( [e_k]_x * P(s) ),
// where P(s) = d(s) R(s) = (1-|s|^2) I - 2 [s]_x + 2 s s^T is an exact
// degree-2 polynomial. Hence the G_poly coefficients follow in closed form
// without any sampling or least-squares fitting.

// B matrix: b = B * vecc(E), rows follow build_rs_system_Ab's residuals
inline MatrixXd build_b_projection_matrix(
    const std::vector<rs::AffineCorrespondence>& acs)
{
    const int N = static_cast<int>(acs.size());
    MatrixXd B(3 * N, 9);
    for (int i = 0; i < N; i++) {
        const Vector3d& q1 = acs[i].q1;
        const Vector3d& q2 = acs[i].q2;
        const Vector3d dq2u = acs[i].dq2_dq1.col(0);
        const Vector3d dq2v = acs[i].dq2_dq1.col(1);
        for (int c = 0; c < 3; c++)
            for (int r = 0; r < 3; r++) {
                const int j = 3 * c + r;
                B(3*i,     j) = q2(r) * q1(c);
                B(3*i + 1, j) = dq2u(r) * q1(c) + ((c == 0) ? q2(r) : 0.0);
                B(3*i + 2, j) = dq2v(r) * q1(c) + ((c == 1) ? q2(r) : 0.0);
            }
    }
    return B;
}

// Coefficients of P(s) = d(s) R(s) over the 10 degree-2 monomials
// (columns ordered as gen_monomials(2)); row j = 3c+r holds entry P(r,c).
inline const Eigen::Matrix<double, 9, 10>& get_cayley_numerator_coeffs() {
    static const Eigen::Matrix<double, 9, 10> Cvec = [] {
        Eigen::Matrix<double, 9, 10> C_;
        C_.setZero();
        const auto idx = build_mono_index(gen_monomials(2));
        auto add = [&](int r, int c, Mono3 m, double v) {
            C_(3 * c + r, idx.at(m)) += v;
        };
        // Diagonal: 1 - |s|^2 + 2 s_i^2
        for (int i = 0; i < 3; i++) {
            add(i, i, {0,0,0}, 1.0);
            for (int j = 0; j < 3; j++) {
                Mono3 m{0,0,0};
                m[j] = 2;
                add(i, i, m, (i == j) ? 1.0 : -1.0);
            }
        }
        // Off-diagonal (r != c): 2 s_r s_c - 2 [s]_x(r,c)
        // [s]_x = {{0,-s3,s2},{s3,0,-s1},{-s2,s1,0}}
        auto add_off = [&](int r, int c, int lin_var, double lin_sign) {
            Mono3 m{0,0,0};
            m[r] += 1; m[c] += 1;
            add(r, c, m, 2.0);
            Mono3 ml{0,0,0};
            ml[lin_var] = 1;
            add(r, c, ml, lin_sign);
        };
        add_off(0, 1, 2, +2.0);  // 2 s1 s2 + 2 s3
        add_off(1, 0, 2, -2.0);  // 2 s1 s2 - 2 s3
        add_off(0, 2, 1, -2.0);  // 2 s1 s3 - 2 s2
        add_off(2, 0, 1, +2.0);  // 2 s1 s3 + 2 s2
        add_off(1, 2, 0, +2.0);  // 2 s2 s3 + 2 s1
        add_off(2, 1, 0, -2.0);  // 2 s2 s3 - 2 s1
        return C_;
    }();
    return Cvec;
}

// Exact G_poly coefficients (replaces the 20-sample fit; identical result
// up to the fit's conditioning error, ~1e-12)
inline GPolyCoeffsDyn fit_G_poly_closed(
    const std::vector<rs::AffineCorrespondence>& acs,
    double /*fy_over_h*/,
    const MatrixXd& U_null)
{
    const int n_null = static_cast<int>(U_null.cols());
    const auto& monoms = gen_monomials(2);

    const MatrixXd B = build_b_projection_matrix(acs);  // 3N×9
    const MatrixXd UB = U_null.transpose() * B;         // n_null×9

    GPolyCoeffsDyn result;
    result.monoms = monoms;
    result.n_rows = n_null;
    result.coeffs.resize(n_null);

    const auto& Cvec = get_cayley_numerator_coeffs();   // 9×10

    for (int k = 0; k < 3; k++) {
        // Mk(:, 3c+m) = sum_r [e_k]_x(r,m) * UB(:, 3c+r)
        const Matrix3d Sk = rs::skew(Vector3d::Unit(k));
        MatrixXd Mk = MatrixXd::Zero(n_null, 9);
        for (int c = 0; c < 3; c++)
            for (int m = 0; m < 3; m++)
                for (int r = 0; r < 3; r++) {
                    const double sv = Sk(r, m);
                    if (sv != 0.0)
                        Mk.col(3*c + m) += sv * UB.col(3*c + r);
                }
        result.coeff_mat[k] = Mk * Cvec;  // n_null×10
        for (int j = 0; j < n_null; j++)
            result.coeffs[j][k] = result.coeff_mat[k].row(j).transpose();
    }

    return result;
}

// Fit G_poly(s) = d(s) * G(s) with degree-2 polynomials, dynamic row count
inline GPolyCoeffsDyn fit_G_poly_dyn(
    const std::vector<rs::AffineCorrespondence>& acs,
    double fy_over_h,
    const MatrixXd& U_null,
    int n_samples = 20,
    unsigned seed = 99)
{
    const int n_null = static_cast<int>(U_null.cols());
    const auto& monoms = gen_monomials(2); // 10 monomials
    const int n_m = static_cast<int>(monoms.size());

    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 0.3);

    MatrixXd V(n_samples, n_m);
    std::vector<MatrixXd> vals(n_samples);

    for (int idx = 0; idx < n_samples; idx++) {
        Vector3d s(dist(rng), dist(rng), dist(rng));
        V.row(idx) = eval_monos(monoms, s).transpose();
        MatrixXd G = build_G_fixed_dyn(s, acs, fy_over_h, U_null);
        double d = 1.0 + s.squaredNorm();
        vals[idx] = d * G;
    }

    GPolyCoeffsDyn result;
    result.monoms = monoms;
    result.n_rows = n_null;
    result.coeffs.resize(n_null);

    auto qr = V.colPivHouseholderQr();
    for (int j = 0; j < n_null; j++) {
        for (int k = 0; k < 3; k++) {
            VectorXd rhs(n_samples);
            for (int idx = 0; idx < n_samples; idx++)
                rhs(idx) = vals[idx](j, k);
            result.coeffs[j][k] = qr.solve(rhs);
        }
    }
    for (int k = 0; k < 3; k++) {
        result.coeff_mat[k].resize(n_null, n_m);
        for (int j = 0; j < n_null; j++)
            result.coeff_mat[k].row(j) = result.coeffs[j][k].transpose();
    }

    return result;
}

// ============================================================
// Left null space of the reference RS system via Householder QR
// ============================================================
// A_ref (3N×12) has full column rank 12 for non-degenerate samples, so the
// last 3N-12 columns of Q span its left null space — the same subspace the
// full SVD in compute_fixed_null_space returns, in a different orthonormal
// basis (the downstream pipeline is invariant to the basis choice).
inline MatrixXd compute_fixed_null_space_qr(
    const std::vector<rs::AffineCorrespondence>& acs,
    double /*fy_over_h*/)
{
    // Inline build of A_ref = build_rs_system_Ab(I, t_ref, acs).  At the
    // reference (R = I, t = t_ref) the twelve dE bases collapse to six
    // constant matrices shared between the (w1,w2) and (v1,v2) blocks:
    //   dE_w1 = -tau1 * [t]x [ek]x,  dE_w2 = tau2 * [t]x [ek]x,
    //   dE_v1 = -tau1 * [ek]x,       dE_v2 = tau2 * [ek]x.
    static const std::array<Matrix3d, 3> base_w = [] {
        const Vector3d t_ref = Vector3d(1.0, 1.0, 1.0) / std::sqrt(3.0);
        std::array<Matrix3d, 3> b;
        for (int k = 0; k < 3; k++)
            b[k] = rs::skew(t_ref) * rs::skew(Vector3d::Unit(k));
        return b;
    }();
    static const std::array<Matrix3d, 3> base_v = [] {
        std::array<Matrix3d, 3> b;
        for (int k = 0; k < 3; k++)
            b[k] = rs::skew(Vector3d::Unit(k));
        return b;
    }();

    const int N = static_cast<int>(acs.size());
    MatrixXd A(3 * N, 12);

    for (int i = 0; i < N; i++) {
        const auto& ac = acs[i];
        const Vector3d& q1 = ac.q1;
        const Vector3d& q2 = ac.q2;
        const double t1 = ac.tau1, t2 = ac.tau2;
        const Vector3d dq2u = ac.dq2_dq1.col(0);
        const Vector3d dq2v = ac.dq2_dq1.col(1);

        for (int k = 0; k < 3; k++) {
            const Matrix3d& Mw = base_w[k];
            const Vector3d Mwq1 = Mw * q1;
            const double w0 = q2.dot(Mwq1);
            const double wu = dq2u.dot(Mwq1) + Mw.col(0).dot(q2);
            const double wv = dq2v.dot(Mwq1) + Mw.col(1).dot(q2);
            A(3*i,     k)     = -t1 * w0;
            A(3*i + 1, k)     = -t1 * wu;
            A(3*i + 2, k)     = -t1 * wv;
            A(3*i,     6 + k) =  t2 * w0;
            A(3*i + 1, 6 + k) =  t2 * wu;
            A(3*i + 2, 6 + k) =  t2 * wv;

            const Matrix3d& Mv = base_v[k];
            const Vector3d Mvq1 = Mv * q1;
            const double v0 = q2.dot(Mvq1);
            const double vu = dq2u.dot(Mvq1) + Mv.col(0).dot(q2);
            const double vv = dq2v.dot(Mvq1) + Mv.col(1).dot(q2);
            A(3*i,     3 + k) = -t1 * v0;
            A(3*i + 1, 3 + k) = -t1 * vu;
            A(3*i + 2, 3 + k) = -t1 * vv;
            A(3*i,     9 + k) =  t2 * v0;
            A(3*i + 1, 9 + k) =  t2 * vu;
            A(3*i + 2, 9 + k) =  t2 * vv;
        }
    }

    const int rows = 3 * N;
    Eigen::HouseholderQR<MatrixXd> qr(A);
    MatrixXd sel = MatrixXd::Zero(rows, rows - 12);
    for (int j = 0; j < rows - 12; j++)
        sel(12 + j, j) = 1.0;
    return qr.householderQ() * sel;  // 3N×(3N-12)
}

// ============================================================
// Polynomial multiplication utilities for minor computation
// ============================================================

// Precomputed multiplication tables for O(1) monomial product lookup.
// mult_2x2[i][j] = index of (m2[i] * m2[j]) in degree-4 basis, or -1
// mult_2x4[i][j] = index of (m2[i] * m4[j]) in degree-6 basis, or -1
struct PolyMultTables {
    int mult_2x2[10][10];   // 10 = C(2+3-1,3-1) degree-2 monomials
    int mult_2x4[10][35];   // 35 = C(4+3-1,3-1) degree-4 monomials

    PolyMultTables() {
        const auto& m2 = gen_monomials(2);
        const auto& m4 = gen_monomials(4);
        const auto& m6 = gen_monomials(6);
        auto m4_idx = build_mono_index(m4);
        auto m6_idx = build_mono_index(m6);

        for (int i = 0; i < 10; i++)
            for (int j = 0; j < 10; j++) {
                Mono3 mp = mono_mult(m2[i], m2[j]);
                auto it = m4_idx.find(mp);
                mult_2x2[i][j] = (it != m4_idx.end()) ? it->second : -1;
            }

        for (int i = 0; i < 10; i++)
            for (int j = 0; j < 35; j++) {
                Mono3 mp = mono_mult(m2[i], m4[j]);
                auto it = m6_idx.find(mp);
                mult_2x4[i][j] = (it != m6_idx.end()) ? it->second : -1;
            }
    }
};

inline const PolyMultTables& get_mult_tables() {
    static const PolyMultTables tables;
    return tables;
}

// Multiply two degree-2 polynomials -> degree-4 polynomial (O(1) lookup).
// Every product of two degree<=2 monomials has degree <=4 and is present in
// the basis, so the table never contains -1; the coefficient vectors are
// dense, so no sparsity test pays off — plain FMA loops vectorize best.
inline void poly_mult_deg2x2_fast(
    const double* c1, const double* c2,
    const int mult_tab[10][10],
    double* result, int /*n4*/)
{
    for (int i = 0; i < 10; i++) {
        const double ci = c1[i];
        for (int j = 0; j < 10; j++)
            result[mult_tab[i][j]] += ci * c2[j];
    }
}

// Multiply degree-2 by degree-4 -> degree-6 polynomial (O(1) lookup)
inline void poly_mult_deg2x4_fast(
    const double* c2, const double* c4,
    const int mult_tab[10][35],
    double* result, int /*n6*/)
{
    for (int i = 0; i < 10; i++) {
        const double ci = c2[i];
        for (int j = 0; j < 35; j++)
            result[mult_tab[i][j]] += ci * c4[j];
    }
}

// ============================================================
// Compute 10 3x3 minors of 5x3 polynomial matrix
// ============================================================
// Each entry is a degree-2 polynomial (10 coefficients).
// Each minor is a degree-6 polynomial (84 coefficients).

inline MatrixXd compute_minor_polys(
    const std::vector<std::array<VectorXd, 3>>& combo_coeffs,
    const std::vector<Mono3>& m2)
{
    constexpr int N2 = 10, N4 = 35, N6 = 84;
    const auto& tabs = get_mult_tables();

    // Enumerate all C(n_rows,3) row combinations
    const int n_rows = static_cast<int>(combo_coeffs.size());
    std::vector<std::array<int,3>> row_combos;
    for (int i = 0; i < n_rows; i++)
        for (int j = i+1; j < n_rows; j++)
            for (int k = j+1; k < n_rows; k++)
                row_combos.push_back({i, j, k});

    const int n_minors = static_cast<int>(row_combos.size());
    MatrixXd minors(n_minors, N6);

    // Shared 2x2 cofactors: minor (r0,r1,r2) expands along row r0 using the
    // three 2x2 minors of row pair (r1,r2); each pair (j,k) with j >= 1 is
    // reused by every minor with r0 < j, so compute them once.
    // cof[pair][0] = cols(1,2), cof[pair][1] = cols(0,2), cof[pair][2] = cols(0,1)
    auto pair_id = [n_rows](int j, int k) { return j * n_rows + k; };
    std::vector<std::array<std::array<double, N4>, 3>> cof(n_rows * n_rows);
    for (int j = 1; j < n_rows; j++) {
        for (int k = j + 1; k < n_rows; k++) {
            const double* Mj0 = combo_coeffs[j][0].data();
            const double* Mj1 = combo_coeffs[j][1].data();
            const double* Mj2 = combo_coeffs[j][2].data();
            const double* Mk0 = combo_coeffs[k][0].data();
            const double* Mk1 = combo_coeffs[k][1].data();
            const double* Mk2 = combo_coeffs[k][2].data();
            auto& c = cof[pair_id(j, k)];
            for (int a = 0; a < 3; a++) c[a].fill(0.0);
            double tmp[N4];

            poly_mult_deg2x2_fast(Mj1, Mk2, tabs.mult_2x2, c[0].data(), N4);
            std::fill(tmp, tmp + N4, 0.0);
            poly_mult_deg2x2_fast(Mj2, Mk1, tabs.mult_2x2, tmp, N4);
            for (int q = 0; q < N4; q++) c[0][q] -= tmp[q];

            poly_mult_deg2x2_fast(Mj0, Mk2, tabs.mult_2x2, c[1].data(), N4);
            std::fill(tmp, tmp + N4, 0.0);
            poly_mult_deg2x2_fast(Mj2, Mk0, tabs.mult_2x2, tmp, N4);
            for (int q = 0; q < N4; q++) c[1][q] -= tmp[q];

            poly_mult_deg2x2_fast(Mj0, Mk1, tabs.mult_2x2, c[2].data(), N4);
            std::fill(tmp, tmp + N4, 0.0);
            poly_mult_deg2x2_fast(Mj1, Mk0, tabs.mult_2x2, tmp, N4);
            for (int q = 0; q < N4; q++) c[2][q] -= tmp[q];
        }
    }

    for (int mi = 0; mi < n_minors; mi++) {
        const int r0 = row_combos[mi][0];
        const int r1 = row_combos[mi][1];
        const int r2 = row_combos[mi][2];

        const double* M00 = combo_coeffs[r0][0].data();
        const double* M01 = combo_coeffs[r0][1].data();
        const double* M02 = combo_coeffs[r0][2].data();
        const auto& c = cof[pair_id(r1, r2)];

        // det = M00*cof(1,2) - M01*cof(0,2) + M02*cof(0,1) (degree 6)
        double det[N6] = {0};
        poly_mult_deg2x4_fast(M00, c[0].data(), tabs.mult_2x4, det, N6);
        { double tmp[N6] = {0}; poly_mult_deg2x4_fast(M01, c[1].data(), tabs.mult_2x4, tmp, N6);
          for (int k = 0; k < N6; k++) det[k] -= tmp[k]; }
        { double tmp[N6] = {0}; poly_mult_deg2x4_fast(M02, c[2].data(), tabs.mult_2x4, tmp, N6);
          for (int k = 0; k < N6; k++) det[k] += tmp[k]; }

        for (int k = 0; k < N6; k++) minors(mi, k) = det[k];
    }

    return minors;
}

// ============================================================
// d(s) division: divide degree-6 minors by d(s) = 1+s1²+s2²+s3²
// ============================================================
// Returns 10×35 matrix of degree-4 polynomial coefficients.
// The division matrix A (84×35) depends only on monomial structure,
// so we precompute and cache its QR factorization.

inline MatrixXd build_ds_division_matrix() {
    const auto& m4 = gen_monomials(4);
    const auto& m6 = gen_monomials(6);
    const int n6 = static_cast<int>(m6.size()); // 84
    const int n4 = static_cast<int>(m4.size()); // 35
    auto m6_idx = build_mono_index(m6);

    // d(s) = 1 + s1^2 + s2^2 + s3^2 — 4 nonzero terms
    struct DTerm { Mono3 m; };
    const DTerm d_terms[] = {
        {{0,0,0}}, {{2,0,0}}, {{0,2,0}}, {{0,0,2}}
    };

    MatrixXd A = MatrixXd::Zero(n6, n4);
    for (int j = 0; j < n4; j++) {
        for (const auto& dt : d_terms) {
            Mono3 mp = mono_mult(dt.m, m4[j]);
            auto it = m6_idx.find(mp);
            if (it != m6_idx.end())
                A(it->second, j) += 1.0;
        }
    }
    return A;
}

inline MatrixXd divide_minors_by_ds(const MatrixXd& minors /* 10×84 */) {
    // Thread-safe lazy initialization (C++11 magic statics).
    // The LS solve against the fixed division matrix is precomputed as an
    // explicit pseudo-inverse so each call is a single small GEMM.
    static const MatrixXd P_div = [] {
        const MatrixXd A_div = build_ds_division_matrix();
        Eigen::ColPivHouseholderQR<MatrixXd> qr_div(A_div);
        return MatrixXd(qr_div.solve(MatrixXd::Identity(A_div.rows(), A_div.rows())));
    }(); // 35×84

    // Q = (A_div \ minors^T)^T = minors * P_div^T = 10×35
    return minors * P_div.transpose();
}

// ============================================================
// Macaulay matrix for degree-4 system at d_ext=5
// ============================================================
// 10 polys × 4 multipliers (degree 1) = 40 rows, 56 columns (degree 5)

inline std::pair<MatrixXd, std::vector<Mono3>> build_macaulay_deg4(
    const MatrixXd& reduced_polys, /* 10×35 */
    int d_ext = 5)
{
    const auto& monoms_deg4 = gen_monomials(4);
    const auto& col_monoms = gen_monomials(d_ext);
    const int n_cols = static_cast<int>(col_monoms.size());

    const int mult_deg = d_ext - 4;
    const auto& mult_monoms = gen_monomials(mult_deg);
    const int n_mult = static_cast<int>(mult_monoms.size());
    const int n_polys = static_cast<int>(reduced_polys.rows());
    const int n_deg4 = static_cast<int>(monoms_deg4.size());

    // Precomputed column-index table: dest_col[mi][j] = column of
    // mult_monoms[mi] * monoms_deg4[j] in the degree-d_ext basis (d_ext=5).
    // Avoids per-call std::map lookups on the hot path.
    static const std::vector<std::vector<int>> dest_col = [] {
        const auto& m4 = gen_monomials(4);
        const auto& m1 = gen_monomials(1);
        const auto idx5 = build_mono_index(gen_monomials(5));
        std::vector<std::vector<int>> tab(m1.size(), std::vector<int>(m4.size(), -1));
        for (size_t a = 0; a < m1.size(); a++)
            for (size_t b = 0; b < m4.size(); b++) {
                auto it = idx5.find(mono_mult(m1[a], m4[b]));
                if (it != idx5.end()) tab[a][b] = it->second;
            }
        return tab;
    }();

    const int n_rows = n_polys * n_mult;
    MatrixXd C = MatrixXd::Zero(n_rows, n_cols);

    if (d_ext == 5) {
        int row = 0;
        for (int pi = 0; pi < n_polys; pi++) {
            for (int mi = 0; mi < n_mult; mi++, row++) {
                const auto& cols = dest_col[mi];
                for (int j = 0; j < n_deg4; j++) {
                    double coeff = reduced_polys(pi, j);
                    if (coeff == 0.0) continue;
                    if (cols[j] >= 0) C(row, cols[j]) += coeff;
                }
            }
        }
        return {C, col_monoms};
    }

    // Generic fallback for other extension degrees
    const auto col_idx = build_mono_index(col_monoms);
    int row = 0;
    for (int pi = 0; pi < n_polys; pi++) {
        for (int mi = 0; mi < n_mult; mi++) {
            for (int j = 0; j < n_deg4; j++) {
                double coeff = reduced_polys(pi, j);
                if (std::abs(coeff) < 1e-18) continue;
                Mono3 mp = mono_mult(mult_monoms[mi], monoms_deg4[j]);
                auto it = col_idx.find(mp);
                if (it != col_idx.end())
                    C(row, it->second) += coeff;
            }
            row++;
        }
    }

    return {C, col_monoms};
}

// ============================================================
// SVD-based action matrix: A_k = V_null^T S_k V_null
// ============================================================

inline MatrixXd build_action_matrix_svd(
    const MatrixXd& V_null,           /* n_cols × n_sol */
    const std::vector<Mono3>& col_monoms,
    const std::map<Mono3, int>& col_idx,
    int var_idx)
{
    const int n_cols = static_cast<int>(V_null.rows());
    const int n_sol = static_cast<int>(V_null.cols());

    // SV = S_k @ V_null where S_k is the shift matrix for variable var_idx
    // S_k[i,j] = 1 iff col_monoms[i] = col_monoms[j] * s_{var_idx}
    MatrixXd SV = MatrixXd::Zero(n_cols, n_sol);
    for (int j = 0; j < n_cols; j++) {
        Mono3 m_new = col_monoms[j];
        m_new[var_idx] += 1;
        auto it = col_idx.find(m_new);
        if (it != col_idx.end())
            SV.row(it->second) += V_null.row(j);
    }

    return V_null.transpose() * SV; // n_sol × n_sol
}

// ============================================================
// Solution structure
// ============================================================

struct DirectSolution {
    Vector3d s;
    double u, v, residual;
};

// ============================================================
// Main algebraic solver
// ============================================================

inline std::vector<DirectSolution> solve_direct(
    const GPolyCoeffsDyn& gpoly,
    unsigned combo_seed = 77,
    int n_combo = 5)
{
    constexpr int D_EXT = 5;
    constexpr int N_SOL = 20;
    constexpr double IMAG_TOL = 0.05;
    constexpr double GAP_TOL = 0.01;

    const int n_rows_G = gpoly.n_rows;
    const auto& monoms_deg2 = gpoly.monoms;
    const int n_m2 = static_cast<int>(monoms_deg2.size());

    // ---- Step 1: Random linear combinations (n_rows_G → n_combo rows) ----
    // The RNG is re-seeded with the same seed on every call, so the weight
    // matrix is a deterministic function of (seed, n_combo, n_rows_G).
    // Cache it thread-locally to avoid re-initializing an mt19937 per call.
    thread_local std::map<std::tuple<unsigned, int, int>, MatrixXd> w_cache;
    const auto w_key = std::make_tuple(combo_seed, n_combo, n_rows_G);
    auto w_it = w_cache.find(w_key);
    if (w_it == w_cache.end()) {
        std::mt19937 rng(combo_seed);
        std::normal_distribution<double> dist(0.0, 1.0);
        MatrixXd W(n_combo, n_rows_G);
        for (int i = 0; i < n_combo; i++)
            for (int j = 0; j < n_rows_G; j++)
                W(i, j) = dist(rng);
        w_it = w_cache.emplace(w_key, std::move(W)).first;
    }
    const MatrixXd& W = w_it->second;

    std::vector<std::array<VectorXd, 3>> combo_coeffs(n_combo);
    for (int k = 0; k < 3; k++) {
        const MatrixXd Ck = W * gpoly.coeff_mat[k];  // n_combo×10
        for (int i = 0; i < n_combo; i++)
            combo_coeffs[i][k] = Ck.row(i).transpose();
    }

    // ---- Step 2: 10 minor polynomials (degree 6) ----
    MatrixXd minors = compute_minor_polys(combo_coeffs, monoms_deg2);

    // ---- Step 3: Divide by d(s) → degree 4 ----
    MatrixXd reduced = divide_minors_by_ds(minors);

    // ---- Step 4: Macaulay matrix (40×56 at d_ext=5) ----
    auto [C, col_monoms] = build_macaulay_deg4(reduced, D_EXT);
    if (C.rows() == 0) return {};

    const int n_cols = static_cast<int>(col_monoms.size());
    if (n_cols <= N_SOL) return {};

    // ---- Step 5: Null space via rank-revealing QR of C^T ----
    // null(C) is the orthogonal complement of colspace(C^T).  A column-
    // pivoted QR of C^T (56×40) exposes the numerical rank on the diagonal
    // of R (the paper-verified sigma_36/sigma_37 > 1e3 gap makes the split
    // unambiguous), and the trailing Q columns give an orthonormal null
    // basis — the same subspace the C^T*C eigendecomposition returns, at a
    // fraction of the cost.
    const int rank_C = n_cols - N_SOL;  // expected numerical rank (36)
    Eigen::ColPivHouseholderQR<MatrixXd> qr_c(C.transpose());
    const auto& qrm = qr_c.matrixQR();

    // Gap check on the R diagonal (|R(k,k)| tracks sigma_k up to a modest
    // factor; compare squared magnitudes like the eigenvalue version did)
    const double d_pivot = qrm(rank_C - 1, rank_C - 1);  // last in-rank pivot
    const double d_null  = qrm(rank_C, rank_C);          // first null pivot
    if (d_null * d_null > d_pivot * d_pivot * GAP_TOL) return {};

    // Trailing N_SOL columns of Q = orthonormal basis of null(C)
    MatrixXd sel_null = MatrixXd::Zero(n_cols, N_SOL);
    for (int j = 0; j < N_SOL; j++)
        sel_null(rank_C + j, j) = 1.0;
    MatrixXd V_null = qr_c.householderQ() * sel_null;  // n_cols × N_SOL

    // ---- Step 6: Action matrix for s1 (20×20, fixed-size) ----
    static const auto col_idx = build_mono_index(gen_monomials(D_EXT));
    const Eigen::Matrix<double, N_SOL, N_SOL> A_s1 =
        build_action_matrix_svd(V_null, col_monoms, col_idx, 0);

    // ---- Step 7: Eigendecompose M_s1^T; recover s2, s3 from eigenvectors ----
    // The coordinate vector of the monomial evaluation, y = V_null^T m(s*),
    // is a LEFT eigenvector of M_s1 (equivalently a right eigenvector of
    // M_s1^T, same spectrum). Its null-space image z = V_null * y equals
    // m(s*) up to scale, so s_k = z[s_k] / z[1]. This replaces the
    // simultaneous diagonalization (complex inverse + two triple products)
    // by reading the same eigenvector data directly.
    using Cd = std::complex<double>;

    Eigen::EigenSolver<Eigen::Matrix<double, N_SOL, N_SOL>> eig(A_s1.transpose());
    const auto& eigenvalues = eig.eigenvalues();
    const auto& V_eig = eig.eigenvectors();

    // Rows of V_null for the monomials {1, s2, s3}
    static const std::array<int, 3> mono_rows = [] {
        const auto idx = build_mono_index(gen_monomials(D_EXT));
        return std::array<int, 3>{idx.at(Mono3{0,0,0}),
                                  idx.at(Mono3{0,1,0}),
                                  idx.at(Mono3{0,0,1})};
    }();
    Eigen::Matrix<double, 3, Eigen::Dynamic> V3(3, N_SOL);
    for (int r = 0; r < 3; r++)
        V3.row(r) = V_null.row(mono_rows[r]);
    const Eigen::Matrix<Cd, 3, Eigen::Dynamic> Z = V3 * V_eig;  // 3×N_SOL

    // ---- Step 8: Extract real solutions, recover (u,v) ----
    std::vector<DirectSolution> solutions;
    solutions.reserve(N_SOL);

    for (int i = 0; i < N_SOL; i++) {
        Cd s1 = eigenvalues(i);
        Cd z0 = Z(0, i);
        // z has unit norm (orthonormal V_null, unit eigenvector); a vanishing
        // constant-monomial entry means a solution at infinity
        if (std::abs(z0) < 1e-12) continue;
        Cd s2 = Z(1, i) / z0;
        Cd s3 = Z(2, i) / z0;

        if (std::abs(s1.imag()) > IMAG_TOL ||
            std::abs(s2.imag()) > IMAG_TOL ||
            std::abs(s3.imag()) > IMAG_TOL)
            continue;

        Vector3d s(s1.real(), s2.real(), s3.real());
        double d_s = 1.0 + s.squaredNorm();
        if (d_s < 0.1) continue;  // Cayley validity

        // Evaluate G(s) for all rows
        VectorXd ms = eval_monos(monoms_deg2, s);
        MatrixXd G(n_rows_G, 3);
        for (int k = 0; k < 3; k++)
            G.col(k) = gpoly.coeff_mat[k] * ms;

        // Solve G[:,1:] * [u,v] = -G[:,0] via normal equations
        MatrixXd A_uv = G.rightCols(2);
        VectorXd rhs = -G.col(0);
        MatrixXd AtA = A_uv.transpose() * A_uv;
        double det_AtA = AtA(0,0) * AtA(1,1) - AtA(0,1) * AtA(1,0);
        if (std::abs(det_AtA) < 1e-20) continue;

        Vector2d uv = AtA.ldlt().solve(A_uv.transpose() * rhs);

        // Compute residual
        VectorXd res_vec = G.col(0) + uv(0) * G.col(1) + uv(1) * G.col(2);
        double residual = res_vec.norm();

        solutions.push_back({s, uv(0), uv(1), residual});
    }

    return solutions;
}

} // namespace rs_direct

// ============================================================
// Solver class: 7-AC Direct (Action Matrix)
// ============================================================

namespace solver {

class RSEssentialMatrixSevenACDirectSolver : public AbstractSolver
{
protected:
    double fy_over_h_;
    double reject_sq_thresh_;  // squared Sampson threshold for verification points
    bool skip_5pt_;            // skip 5pt init for speed (loses early rejection + GS fallback)

public:
    RSEssentialMatrixSevenACDirectSolver(double fy_over_h, double reject_sq_thresh = 0.01, bool skip_5pt = false)
        : fy_over_h_(fy_over_h), reject_sq_thresh_(reject_sq_thresh), skip_5pt_(skip_5pt) {}

    ~RSEssentialMatrixSevenACDirectSolver() {}

    bool returnMultipleModels() const override { return true; }
    size_t maximumSolutions() const override { return 20; }
    size_t sampleSize() const override { return 7; }

    FORCE_INLINE bool estimateModel(
        const DataMatrix& kData_,
        const size_t *kSample_,
        const size_t kSampleNumber_,
        std::vector<models::Model> &models_,
        const double *kWeights_ = nullptr) const override;
};

FORCE_INLINE bool RSEssentialMatrixSevenACDirectSolver::estimateModel(
    const DataMatrix& kData_,
    const size_t *kSample_,
    const size_t kSampleNumber_,
    std::vector<models::Model> &models_,
    const double *kWeights_) const
{
    if (kSampleNumber_ < 7)
        return false;

    const size_t N = kSampleNumber_;

    // Step 1: Extract all 7 ACs
    std::vector<rs::AffineCorrespondence> acs(N);
    std::vector<Eigen::Vector3d> q1s(N), q2s(N);
    for (size_t i = 0; i < N; i++) {
        const size_t idx = (kSample_ ? kSample_[i] : i);
        acs[i] = rs::ac_from_data_row(kData_, idx);
        q1s[i] = acs[i].q1;
        q2s[i] = acs[i].q2;
    }

    // Sanity-check: reject samples that contain NaN/Inf coordinates
    for (size_t i = 0; i < N; i++) {
        if (!acs[i].q1.allFinite() || !acs[i].q2.allFinite())
            return false;
    }

    // Step 2-5: 5pt initialization (provides early rejection, cheirality ref, GS fallback)
    rs::Pose refined;
    bool have_5pt = false;

    if (!skip_5pt_) {
        EssentialMatrixFivePointNisterSolver fivePointSolver;
        std::vector<models::Model> eModels;
        fivePointSolver.estimateModel(kData_, kSample_, 5, eModels);
        if (eModels.empty())
            return false;

        // Select best E using the 6th and 7th point epipolar residuals
        int best_e_idx = 0;
        double best_e_cost = std::numeric_limits<double>::max();
        for (size_t ei = 0; ei < eModels.size(); ei++) {
            Eigen::Matrix3d E = eModels[ei].getData().block<3,3>(0,0);
            double cost = 0;
            for (size_t i = 5; i < N; i++) {
                double r = q2s[i].dot(E.transpose() * q1s[i]);
                cost += r * r;
            }
            if (cost < best_e_cost) {
                best_e_cost = cost;
                best_e_idx = static_cast<int>(ei);
            }
        }

        // Early rejection: if either the 6th or 7th correspondence has high
        // Sampson error with the best 5pt E, skip the expensive algebraic pipeline
        {
            Eigen::Matrix3d E_sel = eModels[best_e_idx].getData().block<3,3>(0,0);
            for (size_t i = 5; i < N; i++) {
                Eigen::Vector3d Etq1 = E_sel.transpose() * q1s[i];
                Eigen::Vector3d Eq2  = E_sel * q2s[i];
                double C = q2s[i].dot(Etq1);
                double denom = Etq1.head<2>().squaredNorm() + Eq2.head<2>().squaredNorm();
                if (denom < 1e-30) return false;
                double sampson_sq = (C * C) / denom;
                if (sampson_sq > reject_sq_thresh_)
                    return false;
            }
        }

        // Decompose best E → (R, t) with cheirality check on all points
        Eigen::Matrix3d E_best = eModels[best_e_idx].getData().block<3,3>(0,0);
        if (!E_best.allFinite())
            return false;
        if (E_best.norm() < 1e-12)
            return false;
        rs::Pose pose = rs::decompose_essential_fast(E_best, q1s, q2s);
        refined = rs::refine_pose_epipolar(pose.R, pose.t, acs);
        have_5pt = true;
    }

    // Step 6: Compute null space at s_ref = 0 (identity rotation)
    // Using s_ref=0 gives better conditioning than s_ref from 5pt.
    Eigen::MatrixXd U_null = rs_direct::compute_fixed_null_space_qr(acs, fy_over_h_);

    // Step 7: Exact G_poly coefficients (9×3 for 7 ACs)
    auto gpoly = rs_direct::fit_G_poly_closed(acs, fy_over_h_, U_null);

    // Step 8: Direct algebraic solver (action matrix)
    auto solutions = rs_direct::solve_direct(gpoly);

    // Sort solutions by polynomial residual (best first)
    std::sort(solutions.begin(), solutions.end(),
              [](const rs_direct::DirectSolution& a, const rs_direct::DirectSolution& b) {
                  return a.residual < b.residual;
              });

    models_.clear();

    // Step 9: Pack models with cheirality check.
    // Complex-conjugate eigenpairs with small imaginary parts pass the
    // IMAG_TOL filter as two near-identical real solutions; packing both
    // only duplicates RANSAC scoring work, so drop exact repeats.
    std::vector<Eigen::Vector3d> packed_s;
    packed_s.reserve(solutions.size());
    for (const auto& sol : solutions) {
        bool duplicate = false;
        for (const auto& sp : packed_s) {
            if ((sol.s - sp).cwiseAbs().maxCoeff() < 1e-6) {
                duplicate = true;
                break;
            }
        }
        if (duplicate)
            continue;
        packed_s.push_back(sol.s);
        Eigen::Matrix3d R_est = rs_eigval::cayley_to_R(sol.s);
        Eigen::Vector3d t_est(1.0, sol.u, sol.v);
        t_est.normalize();

        if (have_5pt) {
            // Use 5pt result for sign disambiguation
            if (t_est.dot(refined.t) < 0) t_est = -t_est;
        }

        // Cheirality check: majority of points must have positive depth
        int pos_depth = 0;
        for (size_t i = 0; i < N; i++) {
            Eigen::Vector3d X_tri = R_est * q1s[i] + t_est;
            if (X_tri(2) > 0 && q1s[i](2) > 0) pos_depth++;
        }
        if (!have_5pt && pos_depth < static_cast<int>(N) / 2) {
            // Without 5pt reference, try flipping t
            t_est = -t_est;
            pos_depth = 0;
            for (size_t i = 0; i < N; i++) {
                Eigen::Vector3d X_tri = R_est * q1s[i] + t_est;
                if (X_tri(2) > 0 && q1s[i](2) > 0) pos_depth++;
            }
        }
        if (pos_depth < static_cast<int>(N) / 2)
            continue;

        // Estimate RS params from this (R, t)
        rs::RSParams rs_est = rs::estimate_rs_params_ac(R_est, t_est, acs, fy_over_h_);

        models::Model model;
        rs::pack_model(model.getMutableData(), R_est, t_est, rs_est);
        models_.emplace_back(std::move(model));

        if (models_.size() >= 20)
            break;
    }

    // GS fallback: only when 5pt is available
    if (have_5pt) {
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
