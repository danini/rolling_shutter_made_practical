// Synthetic benchmark for RS Essential Matrix estimation via SuperANSAC
//
// 5 experiments × 7 methods (ac_rs, pc_rs, gs_5pt, npt_lin, npt_gn, npt_cv, npt_iter):
//   1. Fixed N=100, fixed noise, increasing RS magnitude
//   2. Increasing N, fixed noise, fixed RS
//   3. Fixed N=100, increasing point noise, fixed RS
//   4. Fixed N=100, fixed point noise, increasing affine noise, fixed RS
//   5. Fixed N=100, increasing outlier ratio
//
// Output: CSV to stdout (pipe to file)
//   method, experiment, param_value, trial, R_err, t_err, omega_err, v_err, inlier_count, iterations, time_ms
//
// Usage:
//   ./benchmark_rs_superansac [--method METHOD|all] [--trials N]

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <string>
#include <map>
#include <functional>
#include <memory>
#include <cstring>

// SuperANSAC framework
#include "superansac/include/superansac.h"
#include "superansac/include/samplers/types.h"
#include "superansac/include/scoring/types.h"
#include "superansac/include/local_optimization/types.h"
#include "superansac/include/termination/types.h"

// RS estimator + solvers
#include "superansac/include/estimators/estimator_rs_essential_matrix.h"
#include "superansac/include/estimators/solver_rs_essential_matrix_five_point.h"
#include "superansac/include/estimators/solver_rs_essential_matrix_four_ac.h"
#include "superansac/include/estimators/solver_rs_essential_matrix_bundle_adjustment.h"
#include "superansac/include/estimators/solver_rs_baselines.h"
#include "superansac/include/estimators/solver_rs_essential_matrix_six_ac_eigval.h"
#include "superansac/include/estimators/solver_rs_essential_matrix_seven_ac_direct.h"
#include "superansac/include/estimators/solver_rs_essential_matrix_dai.h"

using namespace Eigen;
using namespace superansac::estimator;
using namespace superansac::estimator::rs;

// ============================================================
// RS projection with iterative tau refinement
// ============================================================
struct Camera {
    Matrix3d K;
    Matrix3d R;
    Vector3d t;
    Vector3d omega;
    Vector3d v;
    int height;
};

bool project_rs(const Camera& cam, const Vector3d& X,
                Vector2d& pixel, double& tau)
{
    Vector3d Xc = cam.R * X + cam.t;
    if (Xc(2) <= 0) return false;

    double h = cam.height;
    Vector2d px_gs;
    px_gs(0) = cam.K(0,0) * Xc(0) / Xc(2) + cam.K(0,2);
    px_gs(1) = cam.K(1,1) * Xc(1) / Xc(2) + cam.K(1,2);

    tau = (px_gs(1) - h / 2.0) / h;
    for (int iter = 0; iter < 5; iter++) {
        Matrix3d R_row = (Matrix3d::Identity() + tau * rs::skew(cam.omega)) * cam.R;
        Vector3d t_row = cam.t + tau * cam.v;
        Vector3d Xc_rs = R_row * X + t_row;
        if (Xc_rs(2) <= 0) return false;
        pixel(0) = cam.K(0,0) * Xc_rs(0) / Xc_rs(2) + cam.K(0,2);
        pixel(1) = cam.K(1,1) * Xc_rs(1) / Xc_rs(2) + cam.K(1,2);
        tau = (pixel(1) - h / 2.0) / h;
    }
    return true;
}

bool compute_ac(const Camera& cam1, const Camera& cam2,
                const Matrix3d& K_inv, int img_w, int img_h,
                const Vector3d& X, AffineCorrespondence& ac)
{
    Vector2d px1, px2;
    double t1, t2;
    if (!project_rs(cam1, X, px1, t1)) return false;
    if (!project_rs(cam2, X, px2, t2)) return false;
    if (px1(0)<0||px1(0)>=img_w||px1(1)<0||px1(1)>=img_h) return false;
    if (px2(0)<0||px2(0)>=img_w||px2(1)<0||px2(1)>=img_h) return false;

    ac.q1 = K_inv * Vector3d(px1(0), px1(1), 1.0);
    ac.q2 = K_inv * Vector3d(px2(0), px2(1), 1.0);
    ac.tau1 = t1;
    ac.tau2 = t2;

    // Numerical Jacobians dq/dX for both cameras
    double eps = 1e-6;
    Matrix3d dq1_dX, dq2_dX;
    for (int k = 0; k < 3; k++) {
        Vector3d Xp = X, Xm = X;
        Xp(k) += eps; Xm(k) -= eps;
        Vector2d px1p, px1m, px2p, px2m;
        double t1p, t1m, t2p, t2m;
        if (!project_rs(cam1, Xp, px1p, t1p)) return false;
        if (!project_rs(cam1, Xm, px1m, t1m)) return false;
        if (!project_rs(cam2, Xp, px2p, t2p)) return false;
        if (!project_rs(cam2, Xm, px2m, t2m)) return false;
        Vector3d q1p = K_inv * Vector3d(px1p(0), px1p(1), 1.0);
        Vector3d q1m = K_inv * Vector3d(px1m(0), px1m(1), 1.0);
        Vector3d q2p = K_inv * Vector3d(px2p(0), px2p(1), 1.0);
        Vector3d q2m = K_inv * Vector3d(px2m(0), px2m(1), 1.0);
        dq1_dX.col(k) = (q1p - q1m) / (2.0 * eps);
        dq2_dX.col(k) = (q2p - q2m) / (2.0 * eps);
    }
    Matrix<double, 2, 3> J1 = dq1_dX.topRows<2>();
    Matrix2d J1J1T = J1 * J1.transpose();
    Matrix<double, 3, 2> pinv_J1 = J1.transpose() * J1J1T.inverse();
    Matrix<double, 2, 3> J2 = dq2_dX.topRows<2>();
    ac.dq2_dq1.setZero();
    ac.dq2_dq1.topRows<2>() = J2 * pinv_J1;
    return true;
}

// ============================================================
// Pack one AC into a DataMatrix row
// ============================================================
void pack_data_row(DataMatrix& data, int row, const AffineCorrespondence& ac)
{
    data(row, 0) = ac.q1(0); data(row, 1) = ac.q1(1);
    data(row, 2) = ac.q2(0); data(row, 3) = ac.q2(1);
    data(row, 4) = ac.tau1;  data(row, 5) = ac.tau2;
    data(row, 6) = ac.dq2_dq1(0,0); data(row, 7) = ac.dq2_dq1(1,0);
    data(row, 8) = ac.dq2_dq1(2,0);
    data(row, 9) = ac.dq2_dq1(0,1); data(row, 10) = ac.dq2_dq1(1,1);
    data(row, 11) = ac.dq2_dq1(2,1);
}

// ============================================================
// Generate synthetic scene + RS data
// ============================================================
struct SceneConfig {
    int N = 100;                  // number of inlier ACs
    double outlier_ratio = 0.3;   // fraction of outliers
    double pt_noise_px = 0.5;     // point noise (pixels)
    double ac_noise = 0.01;       // affine noise (additive on dq2_dq1)
    double rs_scale = 0.5;        // multiplier on RS velocities
};

struct SceneData {
    DataMatrix data;
    Matrix3d R_gt;
    Vector3d t_gt;
    RSParams rs_gt;
    int n_inliers;
};

SceneData generate_scene(const SceneConfig& cfg, std::mt19937& rng)
{
    double fx = 500.0, fy = 500.0;
    int img_w = 640, img_h = 480;
    Matrix3d K;
    K << fx, 0, img_w/2.0,
         0, fy, img_h/2.0,
         0, 0, 1;
    Matrix3d K_inv = K.inverse();

    // Random pose
    std::normal_distribution<double> gauss(0.0, 1.0);
    Vector3d aa_gt(gauss(rng)*0.1, gauss(rng)*0.1, gauss(rng)*0.05);
    Matrix3d R_gt = rs::axis_angle_to_rot(aa_gt);
    Vector3d t_gt(0.3 + gauss(rng)*0.1, -0.1 + gauss(rng)*0.1, 0.9 + gauss(rng)*0.1);
    t_gt.normalize();

    // RS params scaled by rs_scale
    double s = cfg.rs_scale;
    RSParams rs_gt(
        s * Vector3d(0.02, -0.01, 0.015),
        s * Vector3d(0.1, -0.05, 0.2),
        s * Vector3d(-0.01, 0.02, -0.005),
        s * Vector3d(-0.08, 0.15, -0.1)
    );

    Camera cam1, cam2;
    cam1.K = K; cam1.R = Matrix3d::Identity(); cam1.t = Vector3d::Zero();
    cam1.omega = rs_gt.omega1; cam1.v = rs_gt.v1; cam1.height = img_h;
    cam2.K = K; cam2.R = R_gt; cam2.t = t_gt;
    cam2.omega = rs_gt.omega2; cam2.v = rs_gt.v2; cam2.height = img_h;

    int n_total = static_cast<int>(cfg.N / (1.0 - cfg.outlier_ratio));
    int n_outliers = n_total - cfg.N;

    std::uniform_real_distribution<double> dist_xy(-1.5, 1.5);
    std::uniform_real_distribution<double> dist_z(3.0, 8.0);
    std::normal_distribution<double> noise_px(0.0, cfg.pt_noise_px);
    std::normal_distribution<double> noise_ac(0.0, cfg.ac_noise);
    std::uniform_real_distribution<double> unif_norm(-0.5, 0.5);

    DataMatrix data(n_total, 12);
    int valid = 0;

    // Generate inliers
    while (valid < cfg.N) {
        Vector3d X(dist_xy(rng), dist_xy(rng), dist_z(rng));
        AffineCorrespondence ac;
        if (!compute_ac(cam1, cam2, K_inv, img_w, img_h, X, ac))
            continue;

        // Add point noise (in normalized coords)
        if (cfg.pt_noise_px > 0) {
            ac.q1(0) += noise_px(rng) / fx;
            ac.q1(1) += noise_px(rng) / fy;
            ac.q2(0) += noise_px(rng) / fx;
            ac.q2(1) += noise_px(rng) / fy;
        }

        // Add affine noise
        if (cfg.ac_noise > 0) {
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 2; c++)
                    ac.dq2_dq1(r, c) += noise_ac(rng);
        }

        pack_data_row(data, valid, ac);
        valid++;
    }

    // Generate outliers (random correspondences)
    for (int i = 0; i < n_outliers; i++) {
        int row = cfg.N + i;
        // Random normalized coords
        data(row, 0) = unif_norm(rng); data(row, 1) = unif_norm(rng);
        data(row, 2) = unif_norm(rng); data(row, 3) = unif_norm(rng);
        // Random tau in [-0.5, 0.5]
        data(row, 4) = unif_norm(rng); data(row, 5) = unif_norm(rng);
        // Random affine
        for (int k = 6; k < 12; k++)
            data(row, k) = unif_norm(rng);
    }

    // Shuffle rows so outliers are mixed in
    std::vector<int> perm(n_total);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    DataMatrix shuffled(n_total, 12);
    for (int i = 0; i < n_total; i++)
        shuffled.row(i) = data.row(perm[i]);

    return { shuffled, R_gt, t_gt, rs_gt, cfg.N };
}

// ============================================================
// Run SuperANSAC and return errors
// ============================================================
struct Result {
    double R_err_deg;
    double t_err_deg;
    double omega_err;
    double v_err;
    int inlier_count;
    size_t iterations;
    double time_ms;
};

using SolverFactory = std::function<std::pair<
    solver::AbstractSolver*, solver::AbstractSolver*>(double fy_over_h)>;

std::pair<solver::AbstractSolver*, solver::AbstractSolver*>
create_solvers(const std::string& method, double fy_over_h)
{
    if (method == "ac_rs")
        return { new solver::RSEssentialMatrixFivePointSolver(fy_over_h),
                 new solver::PCRSRefinementSolver() };
    if (method == "ac_rs_4")
        return { new solver::RSEssentialMatrixFourACSolver(fy_over_h),
                 new solver::PCRSRefinementSolver() };
    if (method == "ac_rs_6eigval")
        return { new solver::RSEssentialMatrixSixACEigvalSolver(fy_over_h),
                 new solver::PCRSRefinementSolver() };
    if (method == "ac_rs_7direct")
        return { new solver::RSEssentialMatrixSevenACDirectSolver(fy_over_h),
                 new solver::PCRSRefinementSolver() };
    if (method == "dai_20pt")
        return { new solver::Dai20ptSolver(fy_over_h),
                 new solver::LinearRSRefinementSolver() };
    if (method == "dai_44pt")
        return { new solver::Dai44ptSolver(fy_over_h),
                 new solver::GNRSRefinementSolver() };
    if (method == "pc_rs")
        return { new solver::PCFivePointRSSolver(),
                 new solver::PCRSRefinementSolver() };
    if (method == "gs_5pt")
        return { new solver::GSFivePointSolver(),
                 new solver::GSRefinementSolver() };
    if (method == "npt_lin")
        return { new solver::EightPointEssentialSolver(),
                 new solver::LinearRSRefinementSolver() };
    if (method == "npt_gn")
        return { new solver::EightPointEssentialSolver(),
                 new solver::GNRSRefinementSolver() };
    if (method == "npt_cv")
        return { new solver::EightPointEssentialSolver(),
                 new solver::CVRSRefinementSolver() };
    if (method == "npt_iter")
        return { new solver::EightPointEssentialSolver(),
                 new solver::IterativeRSRefinementSolver() };

    std::cerr << "Unknown method: " << method << std::endl;
    return { nullptr, nullptr };
}

Result run_superansac(const SceneData& scene, const std::string& method,
                      double fy_over_h, double threshold,
                      const std::string& config = "msac")
{
    auto [minSolver, nonMinSolver] = create_solvers(method, fy_over_h);
    if (!minSolver || !nonMinSolver) {
        return { 180.0, 180.0, 99.0, 99.0, 0, 0, 0.0 };
    }

    // Create estimator
    auto estimator = std::make_unique<RSEssentialMatrixEstimator>(fy_over_h);
    estimator->setMinimalSolver(minSolver);
    estimator->setNonMinimalSolver(nonMinSolver);

    // Create sampler (Uniform)
    auto sampler = superansac::samplers::createSampler<4>(
        superansac::samplers::SamplerType::Uniform);
    sampler->initialize(scene.data);

    // Create termination criterion
    auto termination = superansac::termination::createTerminationCriterion(
        superansac::termination::TerminationType::RANSAC);
    dynamic_cast<superansac::termination::RANSACCriterion*>(termination.get())
        ->setConfidence(0.99);

    // Configuration-dependent scoring and LO
    superansac::RANSACSettings settings;
    settings.minIterations = 25;
    settings.maxIterations = 1000;
    settings.inlierThreshold = threshold;
    settings.confidence = 0.99;
    settings.sampler = superansac::samplers::SamplerType::Uniform;
    settings.finalOptimization = superansac::local_optimization::LocalOptimizationType::IRLS;
    settings.localOptimizationInsideTheLoop = true;
    settings.useSprt = false;
    settings.topKForLocalOptimization = 3;

    std::unique_ptr<superansac::scoring::AbstractScoring> scorer;
    std::unique_ptr<superansac::local_optimization::LocalOptimizer> localOpt;

    if (config == "magsac") {
        // MAGSAC scoring + LSQ LO
        scorer = superansac::scoring::createScoring<4>(
            superansac::scoring::ScoringType::MAGSAC, false);
        scorer->setThreshold(threshold);
        // MAGSAC needs explicit initialization with DOF
        dynamic_cast<superansac::scoring::MAGSACScoring*>(scorer.get())
            ->initialize(estimator.get());
        settings.scoring = superansac::scoring::ScoringType::MAGSAC;

        localOpt = superansac::local_optimization::createLocalOptimizer(
            superansac::local_optimization::LocalOptimizationType::LSQ);
        dynamic_cast<superansac::local_optimization::LeastSquaresOptimizer*>(localOpt.get())
            ->setUseInliers(true);
        settings.localOptimization = superansac::local_optimization::LocalOptimizationType::LSQ;
    } else {
        // Default: MSAC scoring + LSQ LO
        scorer = superansac::scoring::createScoring<4>(
            superansac::scoring::ScoringType::MSAC, false);
        scorer->setThreshold(threshold);
        settings.scoring = superansac::scoring::ScoringType::MSAC;

        localOpt = superansac::local_optimization::createLocalOptimizer(
            superansac::local_optimization::LocalOptimizationType::LSQ);
        dynamic_cast<superansac::local_optimization::LeastSquaresOptimizer*>(localOpt.get())
            ->setUseInliers(true);
        settings.localOptimization = superansac::local_optimization::LocalOptimizationType::LSQ;
    }

    // Create final optimizer (IRLS)
    auto finalOpt = superansac::local_optimization::createLocalOptimizer(
        superansac::local_optimization::LocalOptimizationType::IRLS);
    auto* irlsOpt = dynamic_cast<superansac::local_optimization::IRLSOptimizer*>(finalOpt.get());
    irlsOpt->setMaxIterations(20);

    // Assemble
    superansac::SupeRansac ransac;
    ransac.setEstimator(estimator.get());
    ransac.setSampler(sampler.get());
    ransac.setScoring(scorer.get());
    ransac.setTerminationCriterion(termination.get());
    ransac.setLocalOptimizer(localOpt.get());
    ransac.setFinalOptimizer(finalOpt.get());
    ransac.setSettings(settings);

    // Run
    auto t0 = std::chrono::high_resolution_clock::now();
    try {
        ransac.run(scene.data);
    } catch (...) {
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return { 180.0, 180.0, 99.0, 99.0, 0, 0, elapsed };
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Extract results
    const auto& inliers = ransac.getInliers();
    if (inliers.size() < 5) {
        return { 180.0, 180.0, 99.0, 99.0, 0, ransac.getIterationNumber(), elapsed };
    }

    Matrix3d R_est;
    Vector3d t_est;
    RSParams rs_est;
    rs::unpack_model(ransac.getBestModel().getData(), R_est, t_est, rs_est);

    double cos_r = ((scene.R_gt.transpose() * R_est).trace() - 1.0) / 2.0;
    double R_err = acos(std::min(1.0, std::max(-1.0, cos_r))) * 180.0 / M_PI;
    double cos_t = std::abs(t_est.normalized().dot(scene.t_gt.normalized()));
    double t_err = acos(std::min(1.0, std::max(-1.0, cos_t))) * 180.0 / M_PI;

    double omega_err = sqrt(
        (rs_est.omega1 - scene.rs_gt.omega1).squaredNorm() +
        (rs_est.omega2 - scene.rs_gt.omega2).squaredNorm());
    double v_err = sqrt(
        (rs_est.v1 - scene.rs_gt.v1).squaredNorm() +
        (rs_est.v2 - scene.rs_gt.v2).squaredNorm());

    return { R_err, t_err, omega_err, v_err,
             static_cast<int>(inliers.size()),
             ransac.getIterationNumber(), elapsed };
}

// ============================================================
// Experiment runner
// ============================================================

void run_experiment(const std::string& method, const std::string& exp_name,
                    const std::string& param_name,
                    const std::vector<std::pair<double, SceneConfig>>& configs,
                    int n_trials, unsigned base_seed,
                    double fy_over_h, double threshold,
                    const std::string& config = "msac")
{
    // Append config suffix for non-default configs
    std::string method_out = method;
    if (config != "msac") method_out = method + "_" + config;

    for (auto& [param_val, cfg_template] : configs) {
        for (int trial = 0; trial < n_trials; trial++) {
            std::mt19937 rng(base_seed + trial);
            SceneConfig cfg = cfg_template;
            SceneData scene = generate_scene(cfg, rng);
            Result res = run_superansac(scene, method, fy_over_h, threshold, config);

            std::cout << method_out << "," << exp_name << "," << param_name << ","
                      << param_val << "," << trial << ","
                      << res.R_err_deg << "," << res.t_err_deg << ","
                      << res.omega_err << "," << res.v_err << ","
                      << res.inlier_count << "," << res.iterations << ","
                      << res.time_ms << std::endl;
        }
        std::cerr << "  [" << method << "/" << exp_name << "] "
                  << param_name << "=" << param_val << " done" << std::endl;
    }
}

// ============================================================
// Main
// ============================================================
int main(int argc, char** argv)
{
    double fy = 500.0;
    int img_h = 480;
    double fy_over_h = fy / img_h;
    double threshold = 0.01;  // Sampson distance threshold (normalized coords)

    int n_trials = 100;
    unsigned base_seed = 12345;

    // Parse args
    std::string method_filter = "all";
    std::string config = "msac";  // "msac" or "magsac"
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--method" && i + 1 < argc)
            method_filter = argv[++i];
        else if (std::string(argv[i]) == "--trials" && i + 1 < argc)
            n_trials = std::atoi(argv[++i]);
        else if (std::string(argv[i]) == "--config" && i + 1 < argc)
            config = argv[++i];
    }

    std::cerr << "Config: " << config
              << " (scoring=" << (config == "magsac" ? "MAGSAC" : "MSAC")
              << ", LO=LSQ)"
              << std::endl;

    std::vector<std::string> all_methods = {
        "ac_rs_7direct", "ac_rs", "ac_rs_4", "pc_rs", "gs_5pt",
        "dai_20pt", "dai_44pt",
        "npt_lin", "npt_gn", "npt_cv", "npt_iter"
    };

    std::vector<std::string> methods;
    if (method_filter == "all") {
        methods = all_methods;
    } else {
        methods.push_back(method_filter);
    }

    // CSV header
    std::cout << "method,experiment,param_name,param_value,trial,"
              << "R_err_deg,t_err_deg,omega_err,v_err,"
              << "inlier_count,iterations,time_ms"
              << std::endl;

    // Build experiment configs
    auto make_cfgs = [](const std::vector<double>& vals, auto setter) {
        std::vector<std::pair<double, SceneConfig>> cfgs;
        for (double v : vals) {
            SceneConfig cfg;
            cfg.N = 100;
            cfg.outlier_ratio = 0.3;
            cfg.pt_noise_px = 0.5;
            cfg.ac_noise = 0.01;
            cfg.rs_scale = 0.5;
            setter(cfg, v);
            cfgs.push_back({v, cfg});
        }
        return cfgs;
    };

    auto exp1 = make_cfgs({0.0, 0.25, 0.5, 1.0, 2.0, 4.0},
        [](SceneConfig& c, double v) { c.rs_scale = v; });

    std::vector<std::pair<double, SceneConfig>> exp2;
    for (int n : {20, 50, 100, 200, 500}) {
        SceneConfig cfg; cfg.N = n; cfg.outlier_ratio = 0.3;
        cfg.pt_noise_px = 0.5; cfg.ac_noise = 0.01; cfg.rs_scale = 0.5;
        exp2.push_back({(double)n, cfg});
    }

    auto exp3 = make_cfgs({0.0, 0.5, 1.0, 2.0, 3.0, 5.0},
        [](SceneConfig& c, double v) { c.pt_noise_px = v; });

    auto exp4 = make_cfgs({0.0, 0.005, 0.01, 0.02, 0.05, 0.1},
        [](SceneConfig& c, double v) { c.ac_noise = v; });

    auto exp5 = make_cfgs({0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9},
        [](SceneConfig& c, double v) { c.outlier_ratio = v; });

    for (const auto& method : methods) {
        std::cerr << "=== Method: " << method << " ===" << std::endl;

        run_experiment(method, "increasing_rs", "rs_scale",
                       exp1, n_trials, base_seed, fy_over_h, threshold, config);
        run_experiment(method, "increasing_N", "N",
                       exp2, n_trials, base_seed, fy_over_h, threshold, config);
        run_experiment(method, "increasing_pt_noise", "pt_noise_px",
                       exp3, n_trials, base_seed, fy_over_h, threshold, config);
        run_experiment(method, "increasing_ac_noise", "ac_noise",
                       exp4, n_trials, base_seed, fy_over_h, threshold, config);
        run_experiment(method, "increasing_outliers", "outlier_ratio",
                       exp5, n_trials, base_seed, fy_over_h, threshold, config);
    }

    std::cerr << "All experiments complete." << std::endl;
    return 0;
}
