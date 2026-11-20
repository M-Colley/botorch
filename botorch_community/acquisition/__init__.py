# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from botorch_community.acquisition.alpha_entropy_search import qAlphaEntropySearch
from botorch_community.acquisition.bayesian_active_learning import (
    qBayesianQueryByComittee,
    qBayesianVarianceReduction,
    qExpectedPredictiveInformationGain,
    qHyperparameterInformedPredictiveExploration,
    qStatisticalDistanceActiveLearning,
)

# NOTE: This import is needed to register the input constructors.
from botorch_community.acquisition.input_constructors import (  # noqa F401
    acqf_input_constructor,
)
from botorch_community.acquisition.local_entropy_search import LocalEntropySearch
from botorch_community.acquisition.lsr_bo import (
    compute_local_box_bounds,
    LocalSearchRolloutAcquisitionFunction,
    LSRBO,
    optimize_lsr_bo_step,
    project_to_local_region,
)
from botorch_community.acquisition.rei import (
    LogRegionalExpectedImprovement,
    qLogRegionalExpectedImprovement,
)
from botorch_community.acquisition.scorebo import qSelfCorrectingBayesianOptimization

__all__ = [
    "compute_local_box_bounds",
    "LocalEntropySearch",
    "LocalSearchRolloutAcquisitionFunction",
    "LogRegionalExpectedImprovement",
    "LSRBO",
    "optimize_lsr_bo_step",
    "project_to_local_region",
    "qAlphaEntropySearch",
    "qBayesianQueryByComittee",
    "qBayesianVarianceReduction",
    "qExpectedPredictiveInformationGain",
    "qHyperparameterInformedPredictiveExploration",
    "qLogRegionalExpectedImprovement",
    "qSelfCorrectingBayesianOptimization",
    "qStatisticalDistanceActiveLearning",
]
