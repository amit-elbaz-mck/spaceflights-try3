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
"""Datasets that log and load artifacts to/from MLRun."""
import logging
import pickle
from typing import Any, Dict, Optional

from kedro.io.core import AbstractDataset
from mlrun.artifacts import get_model
from mlrun.artifacts.base import Artifact
from mlrun.package.context_handler import ContextHandler
from mlrun.package.utils import ArtifactType
from mlrun.package.utils.log_hint_utils import LogHintKey
from mlrun.package.utils.type_hint_utils import TypeHintUtils

from .utils import apply_mlrun, get_current_project, is_applied

logger = logging.getLogger(__name__)


def log_object_with_packagers(
    obj: Any,
    artifact_id: str,
    packing_log_hint: Dict[str, str],
    logging_kwargs: Dict[str, str],
) -> None:
    """Log an object using Packagers.

    Instantiates a ContextHandler and uses it to log the object, which uses
    PackagersManager behind the scenes.

    Args:
        obj: Object to log.
        artifact_id: Id of the artifact that will be logged.
        packing_log_hint: Log hint to pass to `PackagersManager.pack()`, which will
            pass the hint to the selected Packager.
        logging_kwargs: Keyword arguments to pass to `MLClientCtx.log_artifact()`,
            along with the artifact returned from the PackagersManager.
    """
    ctx_handler = ContextHandler()
    # look_for_context loads project's custom packagers
    ctx_handler.look_for_context(args=(), kwargs={})

    log_hint = {LogHintKey.KEY: artifact_id, **packing_log_hint}
    arti = ctx_handler._packagers_manager.pack(obj=obj, log_hint=log_hint)

    if not isinstance(arti, Artifact):
        # objects of builtin types are returned as result, i.e. regular dict, instead
        #   of artifact
        # so, we explicitly pack them as objects
        logger.debug(
            f"Packager did not return an Artifact. Got: {type(arti)}. "
            "Will retry with artifact_type=object"
        )
        log_hint[LogHintKey.ARTIFACT_TYPE] = ArtifactType.OBJECT
        arti = ctx_handler._packagers_manager.pack(obj=obj, log_hint=log_hint)

    assert isinstance(arti, Artifact)
    logger.debug("Type for 'artifact_id=%s' is '%s'", artifact_id, arti.kind)
    ctx_handler._context.log_artifact(item=arti, **logging_kwargs)


class MLRunDataset(AbstractDataset):
    def __init__(
        self,
        artifact_id: str,
        data_type: Optional[str] = None,
        logging_kwargs: Optional[Dict[str, str]] = None,
        packing_log_hint: Optional[Dict[str, str]] = None,
    ) -> None:
        """Instantiate an MLRunDataset.

        Args:
            artifact_id: Id to use when referring to this dataset within the MLRun
                project.
            data_type: Type hint to use when loading back the logged artifact. This
                type determines the Packager class that will be used when loading. If
                not provided, the default MLRun packager will be used for loading.
                `data_type` does not have any affect on logging of the artifact, since
                the actual object type is used at that point.
            logging_kwargs: Keyword arguments to use when logging the artifact.
            packing_log_hint: Log hint to use when packing the artifact. For example,
                `artifact_type` can be set with this hint.
        """
        super().__init__()
        self.artifact_id = artifact_id
        if data_type is None:
            logger.debug(
                "`data_type` was not specified for artifact='%s'. "
                "Unless it was logged using packagers, it cannot be loaded.",
                artifact_id,
            )
        self.data_type = data_type
        self.logging_kwargs = logging_kwargs or {}
        self.packing_log_hint = packing_log_hint or {}

    def _load(self) -> Any:
        ctx_handler = ContextHandler()
        # look_for_context loads project's custom packagers
        ctx_handler.look_for_context(args=(), kwargs={})

        artifact = get_current_project().get_artifact(self.artifact_id)
        di = artifact.to_dataitem()
        packaging_instructions = artifact.spec.unpackaging_instructions

        data_type = self.data_type
        if data_type is not None:
            data_type = TypeHintUtils.parse_type_hint(self.data_type)
            logger.debug("artifact_id=%s, type_cls=%s", self.artifact_id, data_type)

        if packaging_instructions:
            return ctx_handler._packagers_manager._unpack_package(
                data_item=di,
                artifact_key=self.artifact_id,
                packaging_instructions=packaging_instructions,
                type_hint=data_type,
            )

        return ctx_handler._packagers_manager._unpack_data_item(
            data_item=di, type_hint=data_type
        )

    def _save(self, data: Any) -> None:
        # setting db_key is important, otherwise MLRun prepends workflow name
        #   so we end up with dataset name in the catalog and the artifact key
        #   being different
        # note that the user can still override db_key by passing it in logging_kwargs
        log_object_with_packagers(
            obj=data,
            artifact_id=self.artifact_id,
            packing_log_hint=self.packing_log_hint,
            logging_kwargs={"db_key": self.artifact_id, **self.logging_kwargs},
        )

    def _describe(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "data_type": self.data_type,
        }


class MLRunModelDataset(MLRunDataset):
    def _load(self) -> Any:
        arti = get_current_project().get_artifact(self.artifact_id)
        model = self._read_model_artifact(arti)
        apply_mlrun(model, model_path=arti.uri)
        return model

    @staticmethod
    def _read_model_artifact(artifact: Artifact) -> Any:
        temp_path, _spec, _extra_data = get_model(artifact.uri)
        with open(temp_path, "rb") as f:
            return pickle.load(f)

    def _save(self, data: Any) -> None:
        if is_applied(data):
            # re-log the model with the correct name, the name is set to 'model' by
            #   default, which is different than the dataset name
            data._model_handler._model_name = self.artifact_id
            data._model_handler.log()
        else:
            super()._save(data)
