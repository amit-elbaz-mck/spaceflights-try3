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
"""Contains implementations for all callable hooks in the Kedro's execution timeline."""
from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Any, Dict, Set

from kedro.framework.hooks import hook_impl

from .utils import (
    apply_mlrun,
    get_current_project,
    is_applied,
    is_model_dataset_type,
    make_mlrun_model,
)

if TYPE_CHECKING:
    from kedro.io import DataCatalog
    from kedro.pipeline import Pipeline
    from kedro.pipeline.node import Node
    from mlrun.projects import MlrunProject

logger = logging.getLogger(__name__)


class MLRunModelHooks:
    def __init__(self) -> None:
        self._project: MlrunProject | None = None

    @property
    def project(self) -> MlrunProject | None:
        """MLRun project that is used by the Dataset."""
        if not self._project:
            self._project = get_current_project()
        return self._project

    def _get_model_classes(self, catalog: DataCatalog) -> Set:
        classes = set()
        for dataset in catalog.list():
            desc = catalog._data_sets[dataset]._describe()  # noqa: SLF001
            if desc.get("data_type") is None:
                continue

            try:
                module_name, class_name = desc["data_type"].rsplit(sep=".", maxsplit=1)
            except ValueError:
                logger.debug(
                    "Could not find module for data_type %s in dataset %s",
                    desc["data_type"],
                    dataset,
                )
                continue

            try:
                module = importlib.import_module(module_name)
                cls = getattr(module, class_name)
            except (ModuleNotFoundError, AttributeError):
                logger.debug("Could not find data_type for the dataset `%s`", dataset)
                continue

            if is_model_dataset_type(cls):
                classes.add(cls)
        return classes

    @hook_impl
    def before_pipeline_run(  # pylint: disable=unused-argument
        self, run_params: dict[str, Any], pipeline: Pipeline, catalog: DataCatalog
    ) -> None:
        """Hook to be invoked before a pipeline runs."""
        if self.project:
            for cls in self._get_model_classes(catalog):
                make_mlrun_model(cls)

    @hook_impl
    def before_node_run(  # pylint: disable=unused-argument
        self,
        node: Node,
        catalog: DataCatalog,
        inputs: Dict[str, Any],
        is_async: bool,
        session_id: str,
    ) -> Dict[str, Any] | None:
        """Hook to be invoked before a node runs.

        We wrap all objects of any type in `self.classes` with `mlrun_apply`.
        """
        if self.project:
            for inp_name, inp_obj in inputs.items():
                if isinstance(
                    inp_obj, tuple(self._get_model_classes(catalog))
                ) and not is_applied(inp_obj):
                    apply_mlrun(inp_obj, model_name=inp_name)


hooks = MLRunModelHooks()
