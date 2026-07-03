// Python binding implementation for the Rolling-Shutter Essential Matrix estimator.
//
// Input correspondences (Nx12, already in normalized image coordinates):
//   cols 0-1  : x1, y1  (K1^{-1} applied)
//   cols 2-3  : x2, y2  (K2^{-1} applied)
//   cols 4-5  : tau1, tau2  ∈ [-0.5, 0.5]  (normalised row coordinate)
//   cols 6-11 : J flattened col-major (3×2 Jacobian dq2/dq1 in normalised coords)
//              order: J[0,0], J[1,0], J[2,0], J[0,1], J[1,1], J[2,1]
//
// fy_over_h : fy / image_height  (required by RS solver; for TUM-RS ≈ 743/1024 ≈ 0.725)
//
// Returns: (model_8x3, inlier_indices, score, iterations)
//   model_8x3 rows: R(0:3), t^T(3), omega1^T(4), v1^T(5), omega2^T(6), v2^T(7)

#include <Eigen/Dense>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "superansac.h"
#include "settings.h"
#include "samplers/types.h"
#include "scoring/types.h"
#include "termination/types.h"
#include "local_optimization/types.h"
#include "utils/types.h"

#include "estimators/estimator_rs_essential_matrix.h"
#include "estimators/solver_rs_essential_matrix_five_point.h"
#include "estimators/solver_rs_essential_matrix_seven_ac_direct.h"
#include "estimators/solver_rs_essential_matrix_bundle_adjustment.h"
#include "estimators/solver_rs_baselines.h"
#include "estimators/solver_essential_matrix_bundle_adjustment.h"
#include "estimators/solver_rs_essential_matrix_dai.h"

// Template helpers (neighborhood initialization, local optimizer setup).
// Must be included (not forward-declared) so the compiler can instantiate them.
#include "pybind_helpers.h"

// ─── Main function ────────────────────────────────────────────────────────────

std::tuple<Eigen::MatrixXd, std::vector<size_t>, double, size_t>
estimateRSEssentialMatrix(
    const DataMatrix& kCorrespondences_,
    const double kFyOverH_,
    const std::vector<double>& kPointProbabilities_,
    const std::vector<double>& kImageSizes_,
    superansac::RANSACSettings& settings_)
{
    if (kCorrespondences_.cols() != 12)
        throw std::invalid_argument(
            "RS correspondences must have 12 columns "
            "[x1,y1,x2,y2,tau1,tau2,J00,J10,J20,J01,J11,J21].");
    if (kCorrespondences_.rows() < 5)
        throw std::invalid_argument(
            "RS estimator requires at least 5 correspondences.");
    if (kImageSizes_.size() != 4)
        throw std::invalid_argument(
            "image_sizes must have 4 elements "
            "[width_src, height_src, width_dst, height_dst].");
    if (kPointProbabilities_.size() > 0 &&
        kPointProbabilities_.size() != static_cast<size_t>(kCorrespondences_.rows()))
        throw std::invalid_argument(
            "point_probabilities must match the number of correspondences.");

    const superansac::scoring::ScoringType      kScoring          = settings_.scoring;
    const superansac::samplers::SamplerType     kSampler          = settings_.sampler;
    const superansac::neighborhood::NeighborhoodType kNeighborhood = settings_.neighborhood;
    const superansac::local_optimization::LocalOptimizationType kLO  = settings_.localOptimization;
    const superansac::local_optimization::LocalOptimizationType kFO  = settings_.finalOptimization;
    const superansac::termination::TerminationType kTermination       = settings_.terminationCriterion;

    // Build estimator
    auto estimator = std::make_unique<superansac::estimator::RSEssentialMatrixEstimator>(kFyOverH_);
    estimator->setMinimalSolver(
        new superansac::estimator::solver::RSEssentialMatrixFivePointSolver(kFyOverH_));
    estimator->setNonMinimalSolver(
        new superansac::estimator::solver::RSEssentialMatrixBundleAdjustmentSolver(kFyOverH_, 20));

    // Build sampler (sample size = 5)
    auto sampler = superansac::samplers::createSampler<4>(kSampler);

    std::unique_ptr<superansac::neighborhood::AbstractNeighborhoodGraph> neighborhoodGraph;

    if (kSampler == superansac::samplers::SamplerType::PROSAC)
        dynamic_cast<superansac::samplers::PROSACSampler*>(sampler.get())
            ->setSampleSize(estimator->sampleSize());
    else if (kSampler == superansac::samplers::SamplerType::ProgressiveNAPSAC)
    {
        auto* s = dynamic_cast<superansac::samplers::ProgressiveNAPSACSampler<4>*>(sampler.get());
        s->setSampleSize(estimator->sampleSize());
        s->setLayerData({16, 8, 4, 2}, kImageSizes_);
    }
    else if (kSampler == superansac::samplers::SamplerType::NAPSAC)
    {
        initializeNeighborhood<4>(kCorrespondences_, neighborhoodGraph,
                                  kNeighborhood, kImageSizes_, settings_);
        dynamic_cast<superansac::samplers::NAPSACSampler*>(sampler.get())
            ->setNeighborhood(neighborhoodGraph.get());
    }
    else if (kSampler == superansac::samplers::SamplerType::ImportanceSampler)
        dynamic_cast<superansac::samplers::ImportanceSampler*>(sampler.get())
            ->setProbabilities(kPointProbabilities_);
    else if (kSampler == superansac::samplers::SamplerType::ARSampler)
        dynamic_cast<superansac::samplers::AdaptiveReorderingSampler*>(sampler.get())
            ->initialize(&kCorrespondences_, kPointProbabilities_,
                         settings_.arSamplerSettings.estimatorVariance,
                         settings_.arSamplerSettings.randomness);

    sampler->initialize(kCorrespondences_);

    // Build scorer
    auto scorer = superansac::scoring::createScoring<4>(kScoring, settings_.useSprt);
    scorer->setThreshold(settings_.inlierThreshold);

    if (kScoring == superansac::scoring::ScoringType::ACRANSAC)
        scorer->setImageSize(kImageSizes_[0], kImageSizes_[1],
                             kImageSizes_[2], kImageSizes_[3]);
    else if (kScoring == superansac::scoring::ScoringType::MAGSAC)
    {
        if (settings_.useSprt)
            dynamic_cast<superansac::scoring::MAGSACSPRTScoring*>(scorer.get())
                ->initialize(estimator.get());
        else
            dynamic_cast<superansac::scoring::MAGSACScoring*>(scorer.get())
                ->initialize(estimator.get());
    }

    // Build termination criterion
    auto termCriterion = superansac::termination::createTerminationCriterion(kTermination);
    if (kTermination == superansac::termination::TerminationType::RANSAC)
        dynamic_cast<superansac::termination::RANSACCriterion*>(termCriterion.get())
            ->setConfidence(settings_.confidence);

    // Assemble the robust estimator
    superansac::SupeRansac robustEstimator;
    robustEstimator.setEstimator(estimator.get());
    robustEstimator.setSampler(sampler.get());
    robustEstimator.setScoring(scorer.get());
    robustEstimator.setTerminationCriterion(termCriterion.get());

    // Local optimisation (use RS model type)
    std::unique_ptr<superansac::local_optimization::LocalOptimizer> localOptimizer;
    if (kLO != superansac::local_optimization::LocalOptimizationType::None)
    {
        localOptimizer = superansac::local_optimization::createLocalOptimizer(kLO);
        initializeLocalOptimizer<4>(
            kCorrespondences_, localOptimizer, neighborhoodGraph,
            kNeighborhood, kLO, kImageSizes_, settings_,
            settings_.localOptimizationSettings,
            superansac::models::Types::EssentialMatrix, false);
        robustEstimator.setLocalOptimizer(localOptimizer.get());
    }

    // Final optimisation
    std::unique_ptr<superansac::local_optimization::LocalOptimizer> finalOptimizer;
    if (kFO != superansac::local_optimization::LocalOptimizationType::None)
    {
        finalOptimizer = superansac::local_optimization::createLocalOptimizer(kFO);
        initializeLocalOptimizer<4>(
            kCorrespondences_, finalOptimizer, neighborhoodGraph,
            kNeighborhood, kFO, kImageSizes_, settings_,
            settings_.finalOptimizationSettings,
            superansac::models::Types::EssentialMatrix, true);
        robustEstimator.setFinalOptimizer(finalOptimizer.get());
    }

    robustEstimator.setSettings(settings_);
    robustEstimator.run(kCorrespondences_);

    // Empty result on failure
    if (robustEstimator.getInliers().size() < estimator->sampleSize())
        return std::make_tuple(
            Eigen::MatrixXd::Zero(8, 3),
            std::vector<size_t>(),
            0.0,
            robustEstimator.getIterationNumber());

    const Eigen::MatrixXd model = robustEstimator.getBestModel().getData();
    return std::make_tuple(
        model,
        robustEstimator.getInliers(),
        robustEstimator.getBestScore().getValue(),
        robustEstimator.getIterationNumber());
}

// ─── 7-AC Direct Solver variant ───────────────────────────────────────────────
// Uses RSEssentialMatrixSevenACDirectSolver (action matrix) instead of 5pt.
// Sample size = 7.  Debug prints are active while debugging.

std::tuple<Eigen::MatrixXd, std::vector<size_t>, double, size_t>
estimateRSEssentialMatrix7AC(
    const DataMatrix& kCorrespondences_,
    const double kFyOverH_,
    const std::vector<double>& kPointProbabilities_,
    const std::vector<double>& kImageSizes_,
    superansac::RANSACSettings& settings_,
    const double kACWeight_,
    const std::string& kNonMinSolver_)
{
    if (kCorrespondences_.cols() != 12)
        throw std::invalid_argument("RS correspondences must have 12 columns.");
    if (kCorrespondences_.rows() < 7)
        throw std::invalid_argument("7AC estimator requires at least 7 correspondences.");

    const superansac::scoring::ScoringType      kScoring    = settings_.scoring;
    const superansac::samplers::SamplerType     kSampler    = settings_.sampler;
    const superansac::neighborhood::NeighborhoodType kNbhd  = settings_.neighborhood;
    const superansac::local_optimization::LocalOptimizationType kLO = settings_.localOptimization;
    const superansac::local_optimization::LocalOptimizationType kFO = settings_.finalOptimization;
    const superansac::termination::TerminationType kTerm            = settings_.terminationCriterion;

    auto estimator = std::make_unique<superansac::estimator::RSEssentialMatrixEstimator>(kFyOverH_);
    estimator->setMinimalSolver(
        new superansac::estimator::solver::RSEssentialMatrixSevenACDirectSolver(kFyOverH_));
    if (kNonMinSolver_ == "ba")
        estimator->setNonMinimalSolver(
            new superansac::estimator::solver::RSEssentialMatrixBundleAdjustmentSolver(kFyOverH_, 50, kACWeight_));
    else
        estimator->setNonMinimalSolver(
            new superansac::estimator::solver::PCRSRefinementSolver());

    auto sampler = superansac::samplers::createSampler<4>(kSampler);
    std::unique_ptr<superansac::neighborhood::AbstractNeighborhoodGraph> neighborhoodGraph;

    if (kSampler == superansac::samplers::SamplerType::PROSAC)
        dynamic_cast<superansac::samplers::PROSACSampler*>(sampler.get())
            ->setSampleSize(estimator->sampleSize());
    else if (kSampler == superansac::samplers::SamplerType::ImportanceSampler)
        dynamic_cast<superansac::samplers::ImportanceSampler*>(sampler.get())
            ->setProbabilities(kPointProbabilities_);

    sampler->initialize(kCorrespondences_);

    auto scorer = superansac::scoring::createScoring<4>(kScoring, settings_.useSprt);
    scorer->setThreshold(settings_.inlierThreshold);

    if (kScoring == superansac::scoring::ScoringType::MAGSAC)
    {
        if (settings_.useSprt)
            dynamic_cast<superansac::scoring::MAGSACSPRTScoring*>(scorer.get())
                ->initialize(estimator.get());
        else
            dynamic_cast<superansac::scoring::MAGSACScoring*>(scorer.get())
                ->initialize(estimator.get());
    }

    auto termCriterion = superansac::termination::createTerminationCriterion(kTerm);
    if (kTerm == superansac::termination::TerminationType::RANSAC)
        dynamic_cast<superansac::termination::RANSACCriterion*>(termCriterion.get())
            ->setConfidence(settings_.confidence);

    superansac::SupeRansac robustEstimator;
    robustEstimator.setEstimator(estimator.get());
    robustEstimator.setSampler(sampler.get());
    robustEstimator.setScoring(scorer.get());
    robustEstimator.setTerminationCriterion(termCriterion.get());

    std::unique_ptr<superansac::local_optimization::LocalOptimizer> localOptimizer;
    if (kLO != superansac::local_optimization::LocalOptimizationType::None)
    {
        localOptimizer = superansac::local_optimization::createLocalOptimizer(kLO);
        initializeLocalOptimizer<4>(
            kCorrespondences_, localOptimizer, neighborhoodGraph,
            kNbhd, kLO, kImageSizes_, settings_,
            settings_.localOptimizationSettings,
            superansac::models::Types::EssentialMatrix, false);
        robustEstimator.setLocalOptimizer(localOptimizer.get());
    }

    std::unique_ptr<superansac::local_optimization::LocalOptimizer> finalOptimizer;
    if (kFO != superansac::local_optimization::LocalOptimizationType::None)
    {
        finalOptimizer = superansac::local_optimization::createLocalOptimizer(kFO);
        initializeLocalOptimizer<4>(
            kCorrespondences_, finalOptimizer, neighborhoodGraph,
            kNbhd, kFO, kImageSizes_, settings_,
            settings_.finalOptimizationSettings,
            superansac::models::Types::EssentialMatrix, true);
        robustEstimator.setFinalOptimizer(finalOptimizer.get());
    }

    robustEstimator.setSettings(settings_);
    robustEstimator.run(kCorrespondences_);

    if (robustEstimator.getInliers().size() < estimator->sampleSize())
        return std::make_tuple(
            Eigen::MatrixXd::Zero(8, 3),
            std::vector<size_t>(),
            0.0,
            robustEstimator.getIterationNumber());

    const Eigen::MatrixXd model7 = robustEstimator.getBestModel().getData();
    return std::make_tuple(
        model7,
        robustEstimator.getInliers(),
        robustEstimator.getBestScore().getValue(),
        robustEstimator.getIterationNumber());
}

// ─── GS BA wrapper for the RS pipeline ───────────────────────────────────────
// Wraps EssentialMatrixBundleAdjustmentSolver (same BA as GS-5pt) for use in
// the RS estimator's 8×3 model format. Converts 8×3 → 3×3 E, runs poselib BA,
// decomposes the refined E, and packs back as 8×3 with zero RS params.

namespace {
using namespace superansac::estimator::solver;
using superansac::models::Model;
namespace rs = superansac::estimator::rs;

class GSBundleAdjustmentSolver : public AbstractSolver
{
public:
    GSBundleAdjustmentSolver() {}
    ~GSBundleAdjustmentSolver() {}
    bool returnMultipleModels() const override { return false; }
    size_t maximumSolutions() const override { return 1; }
    size_t sampleSize() const override { return 6; }

    FORCE_INLINE bool estimateModel(
        const DataMatrix& kData_,
        const size_t *kSample_,
        const size_t kSampleNumber_,
        std::vector<Model> &models_,
        const double *kWeights_ = nullptr) const override
    {
        if (kSampleNumber_ < sampleSize()) return false;
        if (models_.empty()) return false;

        // Extract R,t from the 8×3 RS model, compute E = [t]× R
        Eigen::Matrix3d R;
        Eigen::Vector3d t;
        rs::RSParams rs_dummy;
        rs::unpack_model(models_[0].getData(), R, t, rs_dummy);

        Eigen::Matrix3d tx;
        tx <<  0, -t(2),  t(1),
               t(2),  0, -t(0),
              -t(1),  t(0),  0;

        // Prepare E as 3×3 model for the BA solver
        std::vector<Model> eModels(1);
        eModels[0].getMutableData() = tx * R;

        // Run the same BA as the GS 5-point estimator (reads only cols 0-3)
        EssentialMatrixBundleAdjustmentSolver baSolver;
        if (!baSolver.estimateModel(kData_, kSample_, kSampleNumber_, eModels, kWeights_))
            return false;
        if (eModels.empty()) return false;

        // Decompose refined E → R,t
        Eigen::Matrix3d E_refined = eModels[0].getData().block<3, 3>(0, 0);
        std::vector<Eigen::Vector3d> q1s, q2s;
        std::vector<double> tau1s, tau2s;
        rs::extract_points_from_data(kData_, kSample_, kSampleNumber_,
                                      q1s, q2s, tau1s, tau2s);
        rs::Pose pose = rs::decompose_essential(E_refined, q1s, q2s);

        // Pack back as 8×3 with zero RS params
        models_.resize(1);
        rs::RSParams rs_zero;
        rs::pack_model(models_[0].getMutableData(), pose.R, pose.t, rs_zero);
        return true;
    }
};
} // anonymous namespace

// ─── Generic RS baseline solver ──────────────────────────────────────────────
// Supports: pc_rs, npt_lin, npt_gn, npt_cv, npt_iter
// Uses RSEssentialMatrixEstimator with method-specific minimal+non-minimal solvers.

static std::pair<
    superansac::estimator::solver::AbstractSolver*,
    superansac::estimator::solver::AbstractSolver*>
create_baseline_solvers(const std::string& method, double fy_over_h)
{
    using namespace superansac::estimator::solver;
    // Dai et al. (CVPR 2016) RS essential matrix solvers
    if (method == "dai_20pt")
        return { new Dai20ptSolver(fy_over_h), new Dai20ptSolver(fy_over_h) };
    if (method == "dai_44pt")
        return { new Dai44ptSolver(fy_over_h), new Dai44ptSolver(fy_over_h) };
    throw std::invalid_argument("Unknown RS baseline method: " + method);
}

std::tuple<Eigen::MatrixXd, std::vector<size_t>, double, size_t>
estimateRSEssentialMatrixGeneric(
    const DataMatrix& kCorrespondences_,
    const std::string& kMethod_,
    const double kFyOverH_,
    const std::vector<double>& kPointProbabilities_,
    const std::vector<double>& kImageSizes_,
    superansac::RANSACSettings& settings_)
{
    if (kCorrespondences_.cols() != 12)
        throw std::invalid_argument(
            "RS correspondences must have 12 columns.");
    if (kImageSizes_.size() != 4)
        throw std::invalid_argument(
            "image_sizes must have 4 elements.");

    auto [minSolver, nonMinSolver] = create_baseline_solvers(kMethod_, kFyOverH_);

    auto estimator = std::make_unique<superansac::estimator::RSEssentialMatrixEstimator>(kFyOverH_);
    estimator->setMinimalSolver(minSolver);
    estimator->setNonMinimalSolver(nonMinSolver);

    const superansac::scoring::ScoringType      kScoring    = settings_.scoring;
    const superansac::samplers::SamplerType     kSampler    = settings_.sampler;
    const superansac::neighborhood::NeighborhoodType kNbhd  = settings_.neighborhood;
    const superansac::local_optimization::LocalOptimizationType kLO = settings_.localOptimization;
    const superansac::local_optimization::LocalOptimizationType kFO = settings_.finalOptimization;
    const superansac::termination::TerminationType kTerm            = settings_.terminationCriterion;

    auto sampler = superansac::samplers::createSampler<4>(kSampler);
    std::unique_ptr<superansac::neighborhood::AbstractNeighborhoodGraph> neighborhoodGraph;

    if (kSampler == superansac::samplers::SamplerType::PROSAC)
        dynamic_cast<superansac::samplers::PROSACSampler*>(sampler.get())
            ->setSampleSize(estimator->sampleSize());
    else if (kSampler == superansac::samplers::SamplerType::ImportanceSampler)
        dynamic_cast<superansac::samplers::ImportanceSampler*>(sampler.get())
            ->setProbabilities(kPointProbabilities_);

    sampler->initialize(kCorrespondences_);

    auto scorer = superansac::scoring::createScoring<4>(kScoring, settings_.useSprt);
    scorer->setThreshold(settings_.inlierThreshold);

    if (kScoring == superansac::scoring::ScoringType::ACRANSAC)
        scorer->setImageSize(kImageSizes_[0], kImageSizes_[1],
                             kImageSizes_[2], kImageSizes_[3]);
    else if (kScoring == superansac::scoring::ScoringType::MAGSAC)
    {
        if (settings_.useSprt)
            dynamic_cast<superansac::scoring::MAGSACSPRTScoring*>(scorer.get())
                ->initialize(estimator.get());
        else
            dynamic_cast<superansac::scoring::MAGSACScoring*>(scorer.get())
                ->initialize(estimator.get());
    }

    auto termCriterion = superansac::termination::createTerminationCriterion(kTerm);
    if (kTerm == superansac::termination::TerminationType::RANSAC)
        dynamic_cast<superansac::termination::RANSACCriterion*>(termCriterion.get())
            ->setConfidence(settings_.confidence);

    superansac::SupeRansac robustEstimator;
    robustEstimator.setEstimator(estimator.get());
    robustEstimator.setSampler(sampler.get());
    robustEstimator.setScoring(scorer.get());
    robustEstimator.setTerminationCriterion(termCriterion.get());

    std::unique_ptr<superansac::local_optimization::LocalOptimizer> localOptimizer;
    if (kLO != superansac::local_optimization::LocalOptimizationType::None)
    {
        localOptimizer = superansac::local_optimization::createLocalOptimizer(kLO);
        initializeLocalOptimizer<4>(
            kCorrespondences_, localOptimizer, neighborhoodGraph,
            kNbhd, kLO, kImageSizes_, settings_,
            settings_.localOptimizationSettings,
            superansac::models::Types::EssentialMatrix, false);
        robustEstimator.setLocalOptimizer(localOptimizer.get());
    }

    std::unique_ptr<superansac::local_optimization::LocalOptimizer> finalOptimizer;
    if (kFO != superansac::local_optimization::LocalOptimizationType::None)
    {
        finalOptimizer = superansac::local_optimization::createLocalOptimizer(kFO);
        initializeLocalOptimizer<4>(
            kCorrespondences_, finalOptimizer, neighborhoodGraph,
            kNbhd, kFO, kImageSizes_, settings_,
            settings_.finalOptimizationSettings,
            superansac::models::Types::EssentialMatrix, true);
        robustEstimator.setFinalOptimizer(finalOptimizer.get());
    }

    robustEstimator.setSettings(settings_);
    robustEstimator.run(kCorrespondences_);

    if (robustEstimator.getInliers().size() < estimator->sampleSize())
        return std::make_tuple(
            Eigen::MatrixXd::Zero(8, 3),
            std::vector<size_t>(),
            0.0,
            robustEstimator.getIterationNumber());

    const Eigen::MatrixXd modelOut = robustEstimator.getBestModel().getData();
    return std::make_tuple(
        modelOut,
        robustEstimator.getInliers(),
        robustEstimator.getBestScore().getValue(),
        robustEstimator.getIterationNumber());
}
