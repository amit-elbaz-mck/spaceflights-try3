# (c) McKinsey & Company 2016 – Present
# All rights reserved
#
#
# This material is intended solely for your internal use and may not be reproduced,
# disclosed or distributed without McKinsey & Company's express prior written consent.
# Except as otherwise stated, the Deliverables are provided ‘as is’, without any express
# or implied warranty, and McKinsey shall not be obligated to maintain, support, host,
# update, or correct the Deliverables. Client guarantees that McKinsey’s use of
# information provided by Client as authorised herein will not violate any law
# or contractual right of a third party. Client is responsible for the operation
# and security of its operating environment. Client is responsible for performing final
# testing (including security testing and assessment) of the code, model validation,
# and final implementation of any model in a production environment. McKinsey is not
# liable for modifications made to Deliverables by anyone other than McKinsey
# personnel, (ii) for use of any Deliverables in a live production environment or
# (iii) for use of the Deliverables by third parties; or
# (iv) the use of the Deliverables for a purpose other than the intended use
# case covered by the agreement with the Client.
# Client warrants that it will not use the Deliverables in a "closed-loop" system,
# including where no Client employee or agent is materially involved in implementing
# the Deliverables and/or insights derived from the Deliverables.
"""Utility methods for interacting with MLRun."""
from __future__ import annotations

import functools
import inspect
import logging
import os
from typing import TYPE_CHECKING, Any, Type, TypeVar, get_args, get_origin

import mlrun
from mlrun.frameworks.sklearn import SKLearnTypes
from mlrun.frameworks.sklearn import apply_mlrun as orig_apply_mlrun
from mlrun.frameworks.sklearn.estimator import Estimator

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from mlrun.projects import MlrunProject

logger = logging.getLogger(__name__)


class MyEstimator(Estimator):
    """The whole purpose of this class is to override `Estimator._calculate_results`.

    By default, _calculate_results expects that x_test and y_test are set for the
    wrapped model, even when there are no metrics to calculate.
    """

    def _calculate_results(
        self,
        y_true: np.ndarray | pd.DataFrame | pd.Series,
        y_pred: np.ndarray | pd.DataFrame | pd.Series,
        is_probabilities: bool,
    ) -> None:
        """Try to calculate results only if there are metrics to calculate."""
        if self._metrics:
            super()._calculate_results(y_true, y_pred, is_probabilities)


Klass = TypeVar("Klass", bound=Type)


def apply_mlrun(model: Any, **kwargs) -> None:
    """Changes the estimator object after `mlrun_apply`."""
    orig_apply_mlrun(model, **kwargs)
    model._mlrun_estimator = MyEstimator()  # noqa: SLF001


def make_mlrun_model(cls: Klass) -> Klass:
    """Override class init to call `mlrun_apply` on self at the end of init."""
    old = cls.__init__

    @functools.wraps(old)
    def fake_init(self, *args, **kwargs) -> None:
        """Apply mlrun on self."""
        old(self, *args, **kwargs)
        apply_mlrun(self)

    cls.__init__ = fake_init
    logger.debug("Wrapped __init__ of class=%s", cls)
    return cls


def is_applied(obj) -> bool:
    return hasattr(obj, "_model_handler")


def get_current_project() -> MlrunProject | None:
    for callstack_frame in inspect.getouterframes(inspect.currentframe()):
        if os.path.join("mlrun", "runtimes", "local") in callstack_frame.filename:
            project_name = mlrun.get_or_create_ctx("context").project
            return mlrun.get_or_create_project(name=project_name)

    return None


MODEL_DATASET_TYPES = get_args(SKLearnTypes.ModelType)


def is_model_dataset_type(typ: Any) -> bool:
    """Check whether a given object can be handled by ModelDataset."""
    if not isinstance(typ, type):
        typ = get_origin(typ)
    return issubclass(typ, MODEL_DATASET_TYPES) if isinstance(typ, type) else False
