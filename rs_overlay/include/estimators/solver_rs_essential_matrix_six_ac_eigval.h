// RS Essential Matrix 6-AC Eigenvalue Solver
//
// Finds RS relative pose via hidden-variable resultant on G(s)*[1,u,v]=0.
//
// Pipeline:
//   1. 5pt Nister → best E → decompose → refine pose
//   2. Cayley s_ref from refined R → fixed left null space of A(R,t)
//   3. Fit G_poly(s): degree-2 polynomial entries (6×3 matrix)
//   4. Eliminate u → 5 equations h_k(s,v) = P_k(s) + v*Q_k(s), deg 4 in s
//   5. Macaulay pencil M(v) = M0 + v*M1 → SVD compress → GEP for v
//   6. For each v: null space → mult matrices → s eigenvalues → enumerate → verify
//   7. Best (s,u,v) → R = cayley_to_R(s), t = [1,u,v]/norm, pack model
//
// Sample size: 6 affine correspondences
#pragma once

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <vector>
#include <array>
#include <map>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <random>
#include <mutex>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "abstract_solver.h"
#include "solver_essential_matrix_five_point_nister.h"
#include "rs_utils.h"
#include "../models/model.h"
#include "../utils/types.h"
#include "../utils/macros.h"

namespace superansac {
namespace estimator {

// ============================================================
// Polynomial algebra utilities for the eigenvalue solver
// ============================================================
namespace rs_eigval {

using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::JacobiSVD;
using Eigen::ComputeFullU;
using Eigen::ComputeFullV;
using Eigen::ComputeThinU;
using Eigen::ComputeThinV;

using Mono3 = std::array<int, 3>;

// Grevlex descending sort key: higher total degree first, then reverse lex
struct GrevlexDesc {
    bool operator()(const Mono3& a, const Mono3& b) const {
        int da = a[0] + a[1] + a[2];
        int db = b[0] + b[1] + b[2];
        if (da != db) return da > db;
        // Reverse lexicographic: compare from last variable
        for (int i = 2; i >= 0; i--) {
            if (a[i] != b[i]) return a[i] < b[i];
        }
        return false;
    }
};

// Generate all monomials in 3 variables with total degree <= max_deg,
// sorted in descending grevlex order. Uses static cache for repeated calls.
inline const std::vector<Mono3>& gen_monomials(int max_deg) {
    static std::map<int, std::vector<Mono3>> cache;
    auto it = cache.find(max_deg);
    if (it != cache.end()) return it->second;

    std::vector<Mono3> monoms;
    for (int d = 0; d <= max_deg; d++) {
        for (int i = d; i >= 0; i--) {
            for (int j = d - i; j >= 0; j--) {
                int k = d - i - j;
                monoms.push_back({i, j, k});
            }
        }
    }
    std::sort(monoms.begin(), monoms.end(), GrevlexDesc());
    cache[max_deg] = std::move(monoms);
    return cache[max_deg];
}

inline Mono3 mono_mult(const Mono3& a, const Mono3& b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

inline double eval_mono(const Mono3& m, const Vector3d& s) {
    double r = 1.0;
    for (int i = 0; i < 3; i++) {
        for (int e = 0; e < m[i]; e++) r *= s(i);
    }
    return r;
}

inline VectorXd eval_monos(const std::vector<Mono3>& monoms, const Vector3d& s) {
    VectorXd v(monoms.size());
    for (size_t i = 0; i < monoms.size(); i++)
        v(i) = eval_mono(monoms[i], s);
    return v;
}

// Build index map from monomial to position
inline std::map<Mono3, int> build_mono_index(const std::vector<Mono3>& monoms) {
    std::map<Mono3, int> idx;
    for (size_t i = 0; i < monoms.size(); i++)
        idx[monoms[i]] = static_cast<int>(i);
    return idx;
}

// ============================================================
// Cayley parameterization
// ============================================================

inline Matrix3d cayley_to_R(const Vector3d& s) {
    Matrix3d S = rs::skew(s);
    Matrix3d I = Matrix3d::Identity();
    return (I - S) * (I + S).inverse();
}

inline Vector3d R_to_cayley(const Matrix3d& R) {
    Matrix3d I = Matrix3d::Identity();
    Matrix3d S = (I - R) * (I + R).inverse();
    return Vector3d(-S(1,2), S(0,2), -S(0,1));
}

// ============================================================
// Build RS linear system A*rs = -b (simplified, no tau corrections)
// ============================================================
// Port of derive_6ac_direct_solver.py:build_rs_system
// 3N×12 system. 12 unknowns: [w1(3), v1(3), w2(3), v2(3)]

inline std::pair<MatrixXd, VectorXd> build_rs_system_Ab(
    const Matrix3d& R, const Vector3d& t,
    const std::vector<rs::AffineCorrespondence>& acs,
    double /*fy_over_h*/)
{
    const int N = static_cast<int>(acs.size());
    MatrixXd A(3 * N, 12);
    VectorXd b(3 * N);
    A.setZero();
    b.setZero();

    Matrix3d E0 = rs::skew(t) * R;
    const Vector3d eu(1, 0, 0), ev(0, 1, 0);

    for (int i = 0; i < N; i++) {
        const auto& ac = acs[i];
        const Vector3d& q1 = ac.q1;
        const Vector3d& q2 = ac.q2;
        double t1 = ac.tau1, t2 = ac.tau2;
        const Vector3d dq2u = ac.dq2_dq1.col(0);
        const Vector3d dq2v = ac.dq2_dq1.col(1);

        // Residuals (constant part)
        b(3*i)     = q2.dot(E0 * q1);
        b(3*i + 1) = dq2u.dot(E0 * q1) + q2.dot(E0 * eu);
        b(3*i + 2) = dq2v.dot(E0 * q1) + q2.dot(E0 * ev);

        for (int k = 0; k < 3; k++) {
            Vector3d ek = Vector3d::Unit(k);

            // w1_k: dE = -tau1 * [t]x R [ek]x
            Matrix3d dE_w1 = -t1 * rs::skew(t) * R * rs::skew(ek);
            A(3*i,     k) = q2.dot(dE_w1 * q1);
            A(3*i + 1, k) = dq2u.dot(dE_w1 * q1) + q2.dot(dE_w1 * eu);
            A(3*i + 2, k) = dq2v.dot(dE_w1 * q1) + q2.dot(dE_w1 * ev);

            // v1_k: dE = [-tau1 R ek]x R
            Matrix3d dE_v1 = rs::skew(-t1 * R * ek) * R;
            A(3*i,     3 + k) = q2.dot(dE_v1 * q1);
            A(3*i + 1, 3 + k) = dq2u.dot(dE_v1 * q1) + q2.dot(dE_v1 * eu);
            A(3*i + 2, 3 + k) = dq2v.dot(dE_v1 * q1) + q2.dot(dE_v1 * ev);

            // w2_k: dE = tau2 * [t]x [ek]x R
            Matrix3d dE_w2 = t2 * rs::skew(t) * rs::skew(ek) * R;
            A(3*i,     6 + k) = q2.dot(dE_w2 * q1);
            A(3*i + 1, 6 + k) = dq2u.dot(dE_w2 * q1) + q2.dot(dE_w2 * eu);
            A(3*i + 2, 6 + k) = dq2v.dot(dE_w2 * q1) + q2.dot(dE_w2 * ev);

            // v2_k: dE = [tau2 ek]x R
            Matrix3d dE_v2 = rs::skew(t2 * ek) * R;
            A(3*i,     9 + k) = q2.dot(dE_v2 * q1);
            A(3*i + 1, 9 + k) = dq2u.dot(dE_v2 * q1) + q2.dot(dE_v2 * eu);
            A(3*i + 2, 9 + k) = dq2v.dot(dE_v2 * q1) + q2.dot(dE_v2 * ev);
        }
    }

    return {A, b};
}

// ============================================================
// Fixed null space computation
// ============================================================

inline MatrixXd compute_fixed_null_space(
    const Vector3d& s_ref,
    const std::vector<rs::AffineCorrespondence>& acs,
    double fy_over_h)
{
    Matrix3d R_ref = cayley_to_R(s_ref);
    Vector3d t_ref = Vector3d(1.0, 1.0, 1.0) / std::sqrt(3.0);
    auto [A_ref, b_ref] = build_rs_system_Ab(R_ref, t_ref, acs, fy_over_h);

    JacobiSVD<MatrixXd> svd(A_ref, ComputeFullU);
    // Left null space: columns 12..17 of U (algebraic rank 12)
    return svd.matrixU().rightCols(A_ref.rows() - 12); // 18×6
}

// ============================================================
// Build G(s) using fixed null space basis
// ============================================================

inline Eigen::Matrix<double, 6, 3> build_G_fixed(
    const Vector3d& s,
    const std::vector<rs::AffineCorrespondence>& acs,
    double fy_over_h,
    const MatrixXd& U_null)
{
    Matrix3d R = cayley_to_R(s);
    Eigen::Matrix<double, 6, 3> G;
    for (int k = 0; k < 3; k++) {
        Vector3d ek = Vector3d::Unit(k);
        auto [Ak, bk] = build_rs_system_Ab(R, ek, acs, fy_over_h);
        G.col(k) = U_null.transpose() * bk;
    }
    return G;
}

// ============================================================
// Fit G_poly(s) = d(s) * G(s), degree-2 polynomial entries
// ============================================================
// Returns: G_coeffs[6][3] arrays of VectorXd (each length 10) and monoms_deg2

struct GPolyCoeffs {
    std::array<std::array<VectorXd, 3>, 6> coeffs; // [row][col] = VectorXd(10)
    std::vector<Mono3> monoms; // 10 degree-2 monomials
};

inline GPolyCoeffs fit_G_poly(
    const std::vector<rs::AffineCorrespondence>& acs,
    double fy_over_h,
    const MatrixXd& U_null,
    int n_samples = 60,
    unsigned seed = 99)
{
    auto monoms = gen_monomials(2); // 10 monomials
    const int n_m = static_cast<int>(monoms.size());

    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 0.3);

    MatrixXd V(n_samples, n_m);
    // vals[sample_idx] = 6x3 matrix
    std::vector<Eigen::Matrix<double, 6, 3>> vals(n_samples);

    for (int idx = 0; idx < n_samples; idx++) {
        Vector3d s(dist(rng), dist(rng), dist(rng));
        V.row(idx) = eval_monos(monoms, s).transpose();
        auto G = build_G_fixed(s, acs, fy_over_h, U_null);
        double d = 1.0 + s.squaredNorm();
        vals[idx] = d * G;
    }

    GPolyCoeffs result;
    result.monoms = monoms;

    for (int j = 0; j < 6; j++) {
        for (int k = 0; k < 3; k++) {
            VectorXd rhs(n_samples);
            for (int idx = 0; idx < n_samples; idx++)
                rhs(idx) = vals[idx](j, k);
            // Least squares: V * c = rhs
            result.coeffs[j][k] = V.colPivHouseholderQr().solve(rhs);
        }
    }

    return result;
}

// ============================================================
// Eliminate u from G(s)*[1,u,v]=0
// ============================================================
// Cross-multiply eq[0] with eq[k] for k=1..5:
//   b[0]*a[k] - a[0]*b[k] + v*(b[0]*c[k] - c[0]*b[k]) = 0
// where a[j], b[j], c[j] are the polynomial coefficients of G_poly[j][0], [j][1], [j][2]

struct UEliminationResult {
    std::vector<VectorXd> P_coeffs; // 5 polynomials, degree 4 in s
    std::vector<VectorXd> Q_coeffs; // 5 polynomials, degree 4 in s
    std::vector<Mono3> monoms_s4;   // degree-4 monomials
};

inline UEliminationResult eliminate_u(const GPolyCoeffs& gpoly) {
    const auto& monoms_s = gpoly.monoms; // degree 2
    auto monoms_s4 = gen_monomials(4);
    const int n_s4 = static_cast<int>(monoms_s4.size());
    auto s4_idx = build_mono_index(monoms_s4);

    // Multiply two polynomials (both in degree-2 basis) -> result in degree-4 basis
    auto poly_mult = [&](const VectorXd& c1, const VectorXd& c2) -> VectorXd {
        VectorXd result = VectorXd::Zero(n_s4);
        for (size_t i = 0; i < monoms_s.size(); i++) {
            if (std::abs(c1(i)) < 1e-18) continue;
            for (size_t j = 0; j < monoms_s.size(); j++) {
                if (std::abs(c2(j)) < 1e-18) continue;
                Mono3 mp = mono_mult(monoms_s[i], monoms_s[j]);
                auto it = s4_idx.find(mp);
                if (it != s4_idx.end())
                    result(it->second) += c1(i) * c2(j);
            }
        }
        return result;
    };

    UEliminationResult res;
    res.monoms_s4 = monoms_s4;

    for (int k = 1; k <= 5; k++) {
        // P_k = b[0]*a[k] - a[0]*b[k]
        VectorXd P_k = poly_mult(gpoly.coeffs[0][1], gpoly.coeffs[k][0])
                      - poly_mult(gpoly.coeffs[0][0], gpoly.coeffs[k][1]);
        // Q_k = b[0]*c[k] - c[0]*b[k]
        VectorXd Q_k = poly_mult(gpoly.coeffs[0][1], gpoly.coeffs[k][2])
                      - poly_mult(gpoly.coeffs[0][2], gpoly.coeffs[k][1]);
        res.P_coeffs.push_back(P_k);
        res.Q_coeffs.push_back(Q_k);
    }

    return res;
}

// ============================================================
// Build Macaulay pencil M(v) = M0 + v*M1
// ============================================================

struct MacaulayPencil {
    MatrixXd M0, M1;
    std::vector<Mono3> col_monoms;
};

inline MacaulayPencil build_macaulay_hidden_v(
    const UEliminationResult& uelim,
    int d_ext)
{
    const auto& P_coeffs = uelim.P_coeffs;
    const auto& Q_coeffs = uelim.Q_coeffs;
    const auto& monoms_s4 = uelim.monoms_s4;
    const int n_eqs = static_cast<int>(P_coeffs.size());

    auto col_monoms = gen_monomials(d_ext);
    const int n_cols = static_cast<int>(col_monoms.size());
    auto col_idx = build_mono_index(col_monoms);

    int mult_max = d_ext - 4;
    if (mult_max < 0) {
        return {MatrixXd::Zero(0, n_cols), MatrixXd::Zero(0, n_cols), col_monoms};
    }
    auto mult_monoms = gen_monomials(mult_max);

    std::vector<VectorXd> M0_rows, M1_rows;
    M0_rows.reserve(n_eqs * mult_monoms.size());
    M1_rows.reserve(n_eqs * mult_monoms.size());

    for (int pi = 0; pi < n_eqs; pi++) {
        for (const auto& mm : mult_monoms) {
            VectorXd row0 = VectorXd::Zero(n_cols);
            VectorXd row1 = VectorXd::Zero(n_cols);
            for (size_t j = 0; j < monoms_s4.size(); j++) {
                Mono3 mp = mono_mult(mm, monoms_s4[j]);
                auto it = col_idx.find(mp);
                if (it == col_idx.end()) continue;
                int idx = it->second;
                if (std::abs(P_coeffs[pi](j)) > 1e-18)
                    row0(idx) += P_coeffs[pi](j);
                if (std::abs(Q_coeffs[pi](j)) > 1e-18)
                    row1(idx) += Q_coeffs[pi](j);
            }
            M0_rows.push_back(row0);
            M1_rows.push_back(row1);
        }
    }

    const int n_rows = static_cast<int>(M0_rows.size());
    MatrixXd M0(n_rows, n_cols), M1(n_rows, n_cols);
    for (int i = 0; i < n_rows; i++) {
        M0.row(i) = M0_rows[i].transpose();
        M1.row(i) = M1_rows[i].transpose();
    }

    return {M0, M1, col_monoms};
}

// ============================================================
// Eigval solution structure
// ============================================================

struct EigvalSolution {
    Vector3d s;
    double u, v, residual;
};

// ============================================================
// Main eigenvalue solver
// ============================================================

inline std::vector<EigvalSolution> solve_eigval(
    const GPolyCoeffs& gpoly,
    int d_ext = 8,
    double sv2_ratio = 1e-2,
    double res_tol = 0.01,
    double sv_tol = 1e-10,
    int max_solutions = 50,
    int max_s_evals = 20,
    int n_threads = 8)
{
    // Limit OMP threads to avoid oversubscription with BLAS
    #ifdef _OPENMP
    omp_set_num_threads(n_threads);
    #endif
    // Step 1: eliminate u
    auto uelim = eliminate_u(gpoly);

    // Step 2: build Macaulay pencil
    auto pencil = build_macaulay_hidden_v(uelim, d_ext);
    const int n_rows = static_cast<int>(pencil.M0.rows());
    const int n_cols = static_cast<int>(pencil.M0.cols());
    if (n_rows == 0) return {};

    // Step 3: SVD-compress to square pencil at a random v
    // Use BDCSVD for large matrices (much faster than JacobiSVD)
    double v_rand = 3.14159;  // fixed value, no need for RNG
    MatrixXd M_rand = pencil.M0 + v_rand * pencil.M1;
    Eigen::BDCSVD<MatrixXd> svd_full(M_rand, ComputeThinU | ComputeThinV);
    auto sv_full = svd_full.singularValues();
    int r_f = 0;
    double sv_thresh = sv_full(0) * sv_tol;
    for (int i = 0; i < sv_full.size(); i++) {
        if (sv_full(i) > sv_thresh) r_f++;
    }

    if (r_f < 2) return {};

    MatrixXd U_r = svd_full.matrixU().leftCols(r_f);
    MatrixXd V_r = svd_full.matrixV().leftCols(r_f);
    MatrixXd M0_sq = U_r.transpose() * pencil.M0 * V_r;
    MatrixXd M1_sq = U_r.transpose() * pencil.M1 * V_r;

    // Step 4: GEP for v. Solve M0*x = -v*M1*x → (-M1)^{-1} M0 x = v x
    Eigen::PartialPivLU<MatrixXd> lu_M1(-M1_sq);
    MatrixXd GEP_matrix = lu_M1.solve(M0_sq);

    Eigen::EigenSolver<MatrixXd> eigsolver(GEP_matrix, false);
    auto all_eigenvalues = eigsolver.eigenvalues();

    // Pre-filter: keep real, finite eigenvalues
    std::vector<double> v_candidates;
    v_candidates.reserve(r_f);
    for (int i = 0; i < all_eigenvalues.size(); i++) {
        double re = all_eigenvalues(i).real();
        double im = all_eigenvalues(i).imag();
        if (std::abs(im) < 0.5 && std::abs(re) < 1e4)
            v_candidates.push_back(re);
    }

    if (v_candidates.empty()) return {};

    // Deduplicate v-candidates
    std::sort(v_candidates.begin(), v_candidates.end());
    {
        std::vector<double> uniq;
        uniq.reserve(v_candidates.size());
        for (double vc : v_candidates) {
            if (uniq.empty() || std::abs(vc - uniq.back()) > 1e-4)
                uniq.push_back(vc);
        }
        v_candidates = std::move(uniq);
    }

    // Score all v-candidates by sv[-2] of compressed pencil (parallel)
    const int n_vcand = static_cast<int>(v_candidates.size());
    std::vector<std::pair<double, double>> scored_v(n_vcand); // (sv2_score, v_value)
    #pragma omp parallel for schedule(dynamic)
    for (int vi = 0; vi < n_vcand; vi++) {
        double vc = v_candidates[vi];
        MatrixXd M_sq = M0_sq + vc * M1_sq;
        Eigen::BDCSVD<MatrixXd> svd_c(M_sq);
        auto svs = svd_c.singularValues();
        int n = static_cast<int>(svs.size());
        double sv2 = (n >= 2) ? svs(n - 2) : svs(n - 1);
        scored_v[vi] = {sv2, vc};
    }

    // Sort by sv2 score (best rank-drop first) — process in this order
    std::sort(scored_v.begin(), scored_v.end());

    // Determine threshold for filtering
    double median_sv2 = scored_v[scored_v.size() / 2].first;
    double threshold = median_sv2 * sv2_ratio;

    // Build sparse shift index maps
    auto col_idx_map = build_mono_index(pencil.col_monoms);
    std::array<std::vector<int>, 3> shift_map;
    for (int var = 0; var < 3; var++) {
        shift_map[var].resize(n_cols, -1);
        Mono3 var_shift = {0, 0, 0};
        var_shift[var] = 1;
        for (int j = 0; j < n_cols; j++) {
            Mono3 m_s = mono_mult(pencil.col_monoms[j], var_shift);
            auto it = col_idx_map.find(m_s);
            if (it != col_idx_map.end())
                shift_map[var][j] = it->second;
        }
    }

    // Step 5: Select which v-candidates to process (threshold + cap)
    std::vector<double> v_to_process;
    v_to_process.reserve(10);
    for (auto& [sv2, vc] : scored_v) {
        if (sv2 >= threshold && static_cast<int>(v_to_process.size()) >= 3)
            break;
        if (static_cast<int>(v_to_process.size()) >= 10) break;
        v_to_process.push_back(vc);
    }

    if (v_to_process.empty()) return {};

    // Process v-candidates in parallel. Each thread collects its own solutions.
    const int n_to_process = static_cast<int>(v_to_process.size());
    std::vector<std::vector<EigvalSolution>> thread_solutions(n_to_process);
    const auto& monoms_s = gpoly.monoms;

    #pragma omp parallel for schedule(dynamic)
    for (int vi = 0; vi < n_to_process; vi++) {
        double v_cand = v_to_process[vi];
        auto& local_sols = thread_solutions[vi];

        // Null space via FULL Macaulay matrix (not compressed).
        // Using the full matrix is critical: the compressed pencil V_r was
        // computed at v_rand, not v_cand, so the null space mapping V_r*N_sq
        // introduces errors that corrupt the multiplication matrices.
        MatrixXd M_full_v = pencil.M0 + v_cand * pencil.M1;
        Eigen::BDCSVD<MatrixXd> svd_full_v(M_full_v, ComputeFullV);
        auto sv_fv = svd_full_v.singularValues();
        int k_null = 0;
        double sv_thresh_v = sv_fv(0) * sv_tol;
        for (int i = static_cast<int>(sv_fv.size()) - 1; i >= 0; i--) {
            if (sv_fv(i) < sv_thresh_v) k_null++;
            else break;
        }
        if (k_null < 1) continue;

        // Null space basis directly in monomial space
        MatrixXd N_basis = svd_full_v.matrixV().rightCols(k_null);

        // Build multiplication matrices using sparse shift maps
        std::array<MatrixXd, 3> M_vars;
        for (int var = 0; var < 3; var++) {
            MatrixXd SN = MatrixXd::Zero(n_cols, k_null);
            for (int j = 0; j < n_cols; j++) {
                int target = shift_map[var][j];
                if (target >= 0)
                    SN.row(target) += N_basis.row(j);
            }
            M_vars[var] = N_basis.transpose() * SN;
        }

        // Get real eigenvalues per variable, sorted by |imag|, capped
        std::array<std::vector<double>, 3> s_evals;
        for (int var = 0; var < 3; var++) {
            Eigen::EigenSolver<MatrixXd> eig_k(M_vars[var], false);
            auto ev_k = eig_k.eigenvalues();
            std::vector<std::pair<double, double>> scored_ev;
            for (int i = 0; i < ev_k.size(); i++) {
                double re = ev_k(i).real();
                double im_abs = std::abs(ev_k(i).imag());
                if (im_abs < 0.1)
                    scored_ev.push_back({im_abs, re});
            }
            std::sort(scored_ev.begin(), scored_ev.end());

            std::vector<double>& real_ev = s_evals[var];
            for (auto& [im, re] : scored_ev) {
                bool dup = false;
                for (double existing : real_ev) {
                    if (std::abs(re - existing) < 1e-4) { dup = true; break; }
                }
                if (!dup) real_ev.push_back(re);
                if (static_cast<int>(real_ev.size()) >= max_s_evals) break;
            }
        }

        // Enumerate (s1,s2,s3) with |s|² < 4 filter
        for (double s1 : s_evals[0]) {
            double s1sq = s1 * s1;
            if (s1sq >= 4.0) continue;
            for (double s2 : s_evals[1]) {
                double s12sq = s1sq + s2 * s2;
                if (s12sq >= 4.0) continue;
                for (double s3 : s_evals[2]) {
                    if (s12sq + s3 * s3 >= 4.0) continue;
                    Vector3d s_est(s1, s2, s3);
                    VectorXd ms = eval_monos(monoms_s, s_est);

                    Eigen::Matrix<double, 6, 1> aa, bb, cc;
                    for (int j = 0; j < 6; j++) {
                        aa(j) = gpoly.coeffs[j][0].dot(ms);
                        bb(j) = gpoly.coeffs[j][1].dot(ms);
                        cc(j) = gpoly.coeffs[j][2].dot(ms);
                    }

                    Eigen::Matrix<double, 6, 1> rhs_u = -(aa + v_cand * cc);
                    double b_norm = bb.squaredNorm();
                    if (b_norm < 1e-20) continue;
                    double u_est = bb.dot(rhs_u) / b_norm;

                    Eigen::Matrix<double, 6, 1> residual = aa + u_est * bb + v_cand * cc;
                    double res_norm = residual.norm();

                    if (res_norm < res_tol) {
                        local_sols.push_back({s_est, u_est, v_cand, res_norm});
                    }
                }
            }
        }
    }

    // Merge thread-local solutions, deduplicate, cap at max_solutions
    std::vector<EigvalSolution> solutions;
    solutions.reserve(max_solutions);
    for (auto& local : thread_solutions) {
        for (auto& sol : local) {
            bool is_dup = false;
            for (const auto& existing : solutions) {
                if ((sol.s - existing.s).squaredNorm() < 1e-6 &&
                    std::abs(sol.u - existing.u) < 1e-3 &&
                    std::abs(sol.v - existing.v) < 1e-3) {
                    is_dup = true;
                    break;
                }
            }
            if (!is_dup) {
                solutions.push_back(std::move(sol));
                if (static_cast<int>(solutions.size()) >= max_solutions)
                    break;
            }
        }
        if (static_cast<int>(solutions.size()) >= max_solutions)
            break;
    }

    return solutions;
}

} // namespace rs_eigval

// ============================================================
// Solver class
// ============================================================

namespace solver {

class RSEssentialMatrixSixACEigvalSolver : public AbstractSolver
{
protected:
    double fy_over_h_;

public:
    RSEssentialMatrixSixACEigvalSolver(double fy_over_h)
        : fy_over_h_(fy_over_h) {}

    ~RSEssentialMatrixSixACEigvalSolver() {}

    bool returnMultipleModels() const override { return true; }
    size_t maximumSolutions() const override { return 20; }
    size_t sampleSize() const override { return 6; }

    FORCE_INLINE bool estimateModel(
        const DataMatrix& kData_,
        const size_t *kSample_,
        const size_t kSampleNumber_,
        std::vector<models::Model> &models_,
        const double *kWeights_ = nullptr) const override;
};

FORCE_INLINE bool RSEssentialMatrixSixACEigvalSolver::estimateModel(
    const DataMatrix& kData_,
    const size_t *kSample_,
    const size_t kSampleNumber_,
    std::vector<models::Model> &models_,
    const double *kWeights_) const
{
    if (kSampleNumber_ < 6)
        return false;

    const size_t N = kSampleNumber_;

    // Step 1: Extract all 6 ACs
    std::vector<rs::AffineCorrespondence> acs(N);
    std::vector<Eigen::Vector3d> q1s(N), q2s(N);
    for (size_t i = 0; i < N; i++) {
        const size_t idx = (kSample_ ? kSample_[i] : i);
        acs[i] = rs::ac_from_data_row(kData_, idx);
        q1s[i] = acs[i].q1;
        q2s[i] = acs[i].q2;
    }

    // Step 2: Run 5pt Nister on the FIRST 5 points only.
    // The 6th point is reserved for selecting among E candidates.
    EssentialMatrixFivePointNisterSolver fivePointSolver;
    std::vector<models::Model> eModels;
    fivePointSolver.estimateModel(kData_, kSample_, 5, eModels);
    if (eModels.empty())
        return false;

    // Step 3: Select best E using the 6th point's epipolar residual
    const Eigen::Vector3d& q1_6 = q1s[5];
    const Eigen::Vector3d& q2_6 = q2s[5];

    int best_e_idx = 0;
    double best_e_cost = std::numeric_limits<double>::max();
    for (size_t ei = 0; ei < eModels.size(); ei++) {
        Eigen::Matrix3d E = eModels[ei].getData().block<3,3>(0,0);
        // Nister stores E transposed
        double r = q2_6.dot(E.transpose() * q1_6);
        double cost = r * r;
        if (cost < best_e_cost) {
            best_e_cost = cost;
            best_e_idx = static_cast<int>(ei);
        }
    }

    // Step 4: Decompose best E → (R, t) with cheirality check on all 6 points
    Eigen::Matrix3d E_best = eModels[best_e_idx].getData().block<3,3>(0,0);
    rs::Pose pose = rs::decompose_essential(E_best, q1s, q2s);

    // Step 5: Refine pose via epipolar GN on all 6 ACs
    rs::Pose refined = rs::refine_pose_epipolar(pose.R, pose.t, acs);

    // Step 6: Cayley parameterization → eigval solver on all 6 ACs
    Eigen::Vector3d s_ref = rs_eigval::R_to_cayley(refined.R);
    Eigen::MatrixXd U_null = rs_eigval::compute_fixed_null_space(s_ref, acs, fy_over_h_);
    auto gpoly = rs_eigval::fit_G_poly(acs, fy_over_h_, U_null);
    // Find many solutions, sorted by polynomial residual.
    // The correct solution tends to have the lowest residual.
    auto solutions = rs_eigval::solve_eigval(gpoly, 8, 1e-2, 0.05, 1e-10, 500, 30);

    // Sort solutions by residual (best first)
    std::sort(solutions.begin(), solutions.end(),
              [](const rs_eigval::EigvalSolution& a, const rs_eigval::EigvalSolution& b) {
                  return a.residual < b.residual;
              });

    models_.clear();

    // Step 7: Return top solutions by residual, with cheirality check.
    for (const auto& sol : solutions) {
        Eigen::Matrix3d R_est = rs_eigval::cayley_to_R(sol.s);
        Eigen::Vector3d t_est(1.0, sol.u, sol.v);
        t_est.normalize();
        if (t_est.dot(refined.t) < 0) t_est = -t_est;

        // Cheirality check: majority of points must have positive depth
        int pos_depth = 0;
        for (size_t i = 0; i < N; i++) {
            Eigen::Vector3d X_tri = R_est * q1s[i] + t_est;
            if (X_tri(2) > 0 && q1s[i](2) > 0) pos_depth++;
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

    // Always include the GS fallback as well
    {
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
