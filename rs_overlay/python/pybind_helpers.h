#pragma once
// Shared template helpers for pybind_functions.cpp and pybind_rs_functions.cpp.
// These must be in a header so each translation unit can instantiate them.

#include <Eigen/Dense>
#include "superansac.h"
#include "samplers/types.h"
#include "scoring/types.h"
#include "local_optimization/types.h"
#include "termination/types.h"
#include "neighborhood/types.h"

using DataMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <size_t _DimensionNumber>
inline void initializeNeighborhood(
    const DataMatrix& kCorrespondences_,
    std::unique_ptr<superansac::neighborhood::AbstractNeighborhoodGraph>& neighborhoodGraph_,
    const superansac::neighborhood::NeighborhoodType kNeighborhoodType_,
    const std::vector<double>& kImageSizes_,
    const superansac::RANSACSettings& kSettings_)
{
    neighborhoodGraph_ = superansac::neighborhood::createNeighborhoodGraph<_DimensionNumber>(kNeighborhoodType_);
    if (kNeighborhoodType_ == superansac::neighborhood::NeighborhoodType::Grid)
    {
        if (kImageSizes_.size() != _DimensionNumber)
            throw std::invalid_argument("The image sizes must have " + std::to_string(_DimensionNumber) + " elements.");
        auto gridNeighborhoodGraph =
            dynamic_cast<superansac::neighborhood::GridNeighborhoodGraph<_DimensionNumber>*>(neighborhoodGraph_.get());
        const auto& kCellNumber = kSettings_.neighborhoodSettings.neighborhoodGridDensity;
        std::vector<double> kCellSizes(_DimensionNumber);
        for (size_t i = 0; i < _DimensionNumber; i++)
        {
            kCellSizes[i] = kImageSizes_[i] / kCellNumber;
            if (kCellSizes[i] < 1.0)
                throw std::invalid_argument("The cell size is too small (< 1px). Try setting a smaller neighborhood size.");
        }
        gridNeighborhoodGraph->setCellSizes(kCellSizes, kCellNumber);
    } else if (kNeighborhoodType_ == superansac::neighborhood::NeighborhoodType::FLANN_KNN)
    {
        auto flannNeighborhoodGraph =
            dynamic_cast<superansac::neighborhood::FlannNeighborhoodGraph<_DimensionNumber, 0>*>(neighborhoodGraph_.get());
        flannNeighborhoodGraph->setNearestNeighborNumber(kSettings_.neighborhoodSettings.nearestNeighborNumber);
    } else if (kNeighborhoodType_ == superansac::neighborhood::NeighborhoodType::FLANN_Radius)
    {
        auto flannNeighborhoodGraph =
            dynamic_cast<superansac::neighborhood::FlannNeighborhoodGraph<_DimensionNumber, 1>*>(neighborhoodGraph_.get());
        flannNeighborhoodGraph->setRadius(kSettings_.neighborhoodSettings.neighborhoodSize);
    }
    neighborhoodGraph_->initialize(&kCorrespondences_);
}

template <size_t _DimensionNumber>
inline void initializeLocalOptimizer(
    const DataMatrix& kCorrespondences_,
    std::unique_ptr<superansac::local_optimization::LocalOptimizer>& localOptimizer_,
    std::unique_ptr<superansac::neighborhood::AbstractNeighborhoodGraph>& neighborhoodGraph_,
    const superansac::neighborhood::NeighborhoodType kNeighborhoodType_,
    const superansac::local_optimization::LocalOptimizationType kLocalOptimizationType_,
    const std::vector<double>& kImageSizes_,
    superansac::RANSACSettings& kSettings_,
    superansac::LocalOptimizationSettings& kLOSettings_,
    superansac::models::Types kModelType_,
    bool kFinalOptimization_ = false)
{
    if (kLocalOptimizationType_ == superansac::local_optimization::LocalOptimizationType::None)
        return;

    if (kLocalOptimizationType_ == superansac::local_optimization::LocalOptimizationType::GCRANSAC)
    {
        if (neighborhoodGraph_ == nullptr)
            initializeNeighborhood<_DimensionNumber>(
                kCorrespondences_, neighborhoodGraph_, kNeighborhoodType_, kImageSizes_, kSettings_);
        auto gcransacLocalOptimizer = dynamic_cast<superansac::local_optimization::GraphCutRANSACOptimizer*>(localOptimizer_.get());
        gcransacLocalOptimizer->setNeighborhood(neighborhoodGraph_.get());
        gcransacLocalOptimizer->setMaxIterations(kLOSettings_.maxIterations);
        gcransacLocalOptimizer->setGraphCutNumber(kLOSettings_.graphCutNumber);
        gcransacLocalOptimizer->setSampleSizeMultiplier(kLOSettings_.sampleSizeMultiplier);
        gcransacLocalOptimizer->setSpatialCoherenceWeight(kLOSettings_.spatialCoherenceWeight);
    } else if (kLocalOptimizationType_ == superansac::local_optimization::LocalOptimizationType::IRLS)
    {
        auto irlsLocalOptimizer = dynamic_cast<superansac::local_optimization::IRLSOptimizer*>(localOptimizer_.get());
        irlsLocalOptimizer->setMaxIterations(kLOSettings_.maxIterations);
        if (kFinalOptimization_ ||
            kModelType_ == superansac::models::Types::Homography ||
            kModelType_ == superansac::models::Types::RigidTransformation ||
            kModelType_ == superansac::models::Types::EssentialMatrix ||
            kModelType_ == superansac::models::Types::FundamentalMatrix)
            irlsLocalOptimizer->setUseInliers(true);
    } else if (kLocalOptimizationType_ == superansac::local_optimization::LocalOptimizationType::LSQ)
    {
        auto lsqLocalOptimizer = dynamic_cast<superansac::local_optimization::LeastSquaresOptimizer*>(localOptimizer_.get());
        if (kFinalOptimization_ ||
            kModelType_ == superansac::models::Types::Homography ||
            kModelType_ == superansac::models::Types::RigidTransformation ||
            kModelType_ == superansac::models::Types::EssentialMatrix ||
            kModelType_ == superansac::models::Types::FundamentalMatrix)
            lsqLocalOptimizer->setUseInliers(true);
    } else if (kLocalOptimizationType_ == superansac::local_optimization::LocalOptimizationType::NestedRANSAC)
    {
        auto nestedRansacLocalOptimizer = dynamic_cast<superansac::local_optimization::NestedRANSACOptimizer*>(localOptimizer_.get());
        nestedRansacLocalOptimizer->setMaxIterations(kLOSettings_.maxIterations);
        nestedRansacLocalOptimizer->setSampleSizeMultiplier(kLOSettings_.sampleSizeMultiplier);
    } else if (kLocalOptimizationType_ == superansac::local_optimization::LocalOptimizationType::IteratedLMEDS)
    {
        auto iteratedLMEDSLocalOptimizer = dynamic_cast<superansac::local_optimization::IteratedLMEDSOptimizer*>(localOptimizer_.get());
        iteratedLMEDSLocalOptimizer->setModelType(kModelType_);
    } else if (kLocalOptimizationType_ == superansac::local_optimization::LocalOptimizationType::CrossValidation)
    {
        auto crossValidationLocalOptimizer = dynamic_cast<superansac::local_optimization::CrossValidationOptimizer*>(localOptimizer_.get());
        if (kFinalOptimization_ ||
            kModelType_ == superansac::models::Types::Homography ||
            kModelType_ == superansac::models::Types::RigidTransformation ||
            kModelType_ == superansac::models::Types::EssentialMatrix ||
            kModelType_ == superansac::models::Types::FundamentalMatrix)
            crossValidationLocalOptimizer->setUseInliers(true);
        crossValidationLocalOptimizer->setSampleSizeMultiplier(kLOSettings_.sampleSizeMultiplier);
    }
}
