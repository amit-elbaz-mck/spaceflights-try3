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
import inspect
import logging
import re
import shutil
import string
import subprocess
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, get_origin, get_type_hints

import click
import dotenv
import jinja2
import mlrun
import yaml
from kedro.framework.project import pipelines
from kedro.framework.session import KedroSession
from kedro.framework.startup import ProjectMetadata
from kedro.pipeline import Pipeline
from slugify import slugify

from kedro_mlrun.utils import is_model_dataset_type

if typing.TYPE_CHECKING:
    from mlrun.projects import MlrunProject

TEMPLATES_DIR = Path(__file__).parent / "templates"

logger = logging.getLogger(__name__)


def typehint_to_str(hint: Any) -> str:
    """Get string representation of the type hint."""
    if isinstance(hint, str):
        return hint

    if isinstance(hint, type):
        return f"{hint.__module__}.{hint.__name__}"

    return str(hint)


@dataclass
class MLRunFunction:
    name: str
    run_args: List[str]
    run_after: List[str]


class MLRunWorkflow(ABC):
    def __init__(self, env: str = "local") -> None:
        self._env = env

    @property
    @abstractmethod
    def datasets(self) -> List[str]:
        """All datasets that must be persisted to run the workflow."""

    @property
    @abstractmethod
    def free_inputs(self) -> Set[str]:
        """All datasets that must be persisted before the workflow can run."""

    @property
    @abstractmethod
    def functions(self) -> List[MLRunFunction]:
        """All functions that make up the workflow."""

    @property
    def env(self) -> str:
        """Kedro environment."""
        return self._env


class MLRunNodeWorkflow(MLRunWorkflow):
    def __init__(self, env: str, pipe: Pipeline) -> None:
        super().__init__(env)
        deps = pipe.node_dependencies
        nodes = pipe.nodes
        self._free_inputs = pipe.inputs()
        self._functions = [
            MLRunFunction(
                name=node.name,
                run_args=["-n", node.name],
                run_after=[n.name for n in deps[node]],
            )
            for node in nodes
        ]

        # ordering important here to keep catalog looking human-readable
        self._datasets = []
        for node in nodes:
            self._datasets.extend(
                dataset
                for dataset in node.inputs
                if (
                    not dataset.startswith("params:")
                    and dataset != "parameters"
                    and dataset not in self._datasets
                )
            )

            self._datasets.extend(
                dataset
                for dataset in node.outputs
                if (
                    not dataset.startswith("params:")
                    and dataset != "parameters"
                    and dataset not in self._datasets
                )
            )

    @property
    def datasets(self) -> List[str]:
        return self._datasets

    @property
    def free_inputs(self) -> Set[str]:
        return self._free_inputs

    @property
    def functions(self) -> List[MLRunFunction]:
        return self._functions


@click.group(name="kedro-mlrun")
def cli() -> None:
    """Kedro mlrun."""


@cli.group(name="mlrun")
def mlrun_group() -> None:
    """Kedro mlrun."""


@mlrun_group.command()
@click.option("-p", "--pipeline", "pipeline_name", default="__default__")
@click.option("-e", "--env", "env", default="local")
@click.option("-b", "--build", "build", default=False)
@click.pass_obj
def init(metadata: ProjectMetadata, pipeline_name: str, env: str, build: bool) -> None:
    """Initialize MLRun features for this kedro project."""
    wf = MLRunNodeWorkflow(env=env, pipe=pipelines[pipeline_name])
    write_kedro_handler(metadata)
    write_workflow(metadata, workflow=wf)
    write_project_setups(metadata)
    write_runner(metadata)
    create_env(metadata, workflow=wf)
    add_hook(metadata)
    (metadata.project_path / ".mlrun" / ".gitignore").write_text("mlrun.env\n")


@mlrun_group.command()
@click.option("-p", "--pipeline", "pipeline_name", default="__default__")
@click.option("-e", "--env", default="local")
@click.pass_obj
def rebuild(metadata: ProjectMetadata, pipeline_name: str, env: str) -> None:
    """Rebuild MLRun files from kedro project."""
    wf = MLRunNodeWorkflow(env=env, pipe=pipelines[pipeline_name])
    write_kedro_handler(metadata)
    write_workflow(metadata, workflow=wf)
    write_save(metadata)
    write_runner(metadata)
    create_env(metadata, workflow=wf)


@mlrun_group.command()
@click.pass_obj
def save(metadata: ProjectMetadata) -> None:
    """Push project to remote environment."""
    subprocess.run(
        ["python", metadata.project_path / ".mlrun" / "save.py"],
        stdout=None,
        stderr=subprocess.STDOUT,
        check=True,
        cwd=str(metadata.project_path),
        text=True,
    )


@mlrun_group.command()
@click.pass_obj
def run(metadata: ProjectMetadata) -> None:
    """Run project on remote Iguazio environment."""
    subprocess.run(
        ["python", metadata.project_path / "mlrun_runner.py"],
        stdout=None,
        stderr=subprocess.STDOUT,
        check=True,
        cwd=str(metadata.project_path),
        text=True,
    )


@mlrun_group.command(no_args_is_help=True)
@click.option("-d", "--mlrun-dbpath")
@click.option("-u", "--v3io-username")
@click.option("-a", "--v3io-access-key")
@click.pass_obj
def config(
    metadata: ProjectMetadata,
    mlrun_dbpath: str,
    v3io_username: str,
    v3io_access_key: str,
) -> None:
    """Set config value for MLRun."""
    conf_file = metadata.project_path / ".mlrun" / "mlrun.env"

    if mlrun_dbpath:
        dotenv.set_key(conf_file, "MLRUN_DBPATH", mlrun_dbpath)
    if v3io_username:
        dotenv.set_key(conf_file, "V3IO_USERNAME", v3io_username)
    if v3io_access_key:
        dotenv.set_key(conf_file, "V3IO_ACCESS_KEY", v3io_access_key)


@mlrun_group.command()
@click.option("-p", "--pipeline", "pipeline_name", default="__default__")
@click.option("-e", "--env", default="local")
@click.pass_obj
def log_inputs(metadata: ProjectMetadata, pipeline_name: str, env: str) -> None:
    """Log global free inputs."""
    session = KedroSession.create(env=env)
    catalog = session.load_context().catalog
    global_free_inputs = pipelines[pipeline_name].inputs()

    mlrun.set_env_from_file(str(metadata.project_path / ".mlrun" / "mlrun.env"))
    project: MlrunProject = mlrun.get_or_create_project(
        name=to_project_name(metadata.project_name), context="./", user_project=True
    )

    for dataset in global_free_inputs:
        if not dataset.startswith("params:") and dataset != "parameters":
            if not hasattr(catalog._data_sets[dataset], "_filepath"):  # noqa: SLF001
                logger.warning(
                    "Not logging dataset `%s`. It does not have a _filepath attribute.",
                    dataset,
                )
            fp = str(catalog._data_sets[dataset]._filepath)  # noqa: SLF001
            # local_path in log_artifact() can actually be a local or remote URI.
            #  If local, the file will be uploaded to the MLRun artifact store
            # mlrun does not complain if the file to be uploaded does not exist, and
            #   uploads a zero-byte file instead. so, we check the file ourselves
            is_local = fp.startswith("/") or "://" not in fp
            if is_local and not Path(fp).exists():
                raise ValueError(f"Local input does not exist: {fp}")
            project.log_artifact(dataset, local_path=fp)
            # we also set the artifact in the project, so that it is logged in
            # project.yaml
            project.set_artifact(dataset, target_path=project.get_artifact(dataset).target_path)

    project.save()


def write_kedro_handler(metadata: ProjectMetadata) -> None:
    (metadata.project_path / ".mlrun").mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        src=TEMPLATES_DIR / "kedro_handler.py.j2",
        dst=metadata.project_path / ".mlrun" / "kedro_handler.py",
    )


def write_workflow(metadata: ProjectMetadata, workflow: MLRunWorkflow) -> None:
    (metadata.project_path / ".mlrun").mkdir(parents=True, exist_ok=True)
    handlers_template = create_template(TEMPLATES_DIR / "workflow.py.j2")
    handlers_template.stream(
        project_name=to_project_name(metadata.project_name),
        env=f"{workflow.env}_mlrun",
        # functions: List[key: str, key_as_var: str, after: List[str], args: List[str]]
        functions=[
            (
                to_mlrun_func_name(func.name),
                to_variable_name(func.name),
                [to_variable_name(after_name) for after_name in func.run_after],
                func.run_args,
            )
            for func in workflow.functions
        ],
    ).dump(str(metadata.project_path / ".mlrun" / "workflow.py"))


def write_save(metadata: ProjectMetadata) -> None:
    (metadata.project_path / ".mlrun").mkdir(parents=True, exist_ok=True)
    handlers_template = create_template(TEMPLATES_DIR / "save.py.j2")
    handlers_template.stream(project_name=to_project_name(metadata.project_name)).dump(
        str(metadata.project_path / ".mlrun" / "save.py")
    )

def write_project_setups(metadata: ProjectMetadata) -> None:
    (metadata.project_path / ".mlrun").mkdir(parents=True, exist_ok=True)
    project_setup_template = create_template(TEMPLATES_DIR / "project_setup.py.j2")
    project_setup_template.stream(project_name=to_project_name(metadata.project_name)).dump(
        str(metadata.project_path / "project_setup.py")
    )

    project_setup_extras_template = create_template(TEMPLATES_DIR / "project_setup_extras.py.j2")
    project_setup_extras_template.stream(project_name=to_project_name(metadata.project_name)).dump(
        str(metadata.project_path / "project_setup_extras.py")
    )


def write_runner(metadata: ProjectMetadata) -> None:
    handlers_template = create_template(TEMPLATES_DIR / "mlrun_runner.py.j2")
    handlers_template.stream(project_name=to_project_name(metadata.project_name)).dump(
        str(metadata.project_path / "mlrun_runner.py")
    )


def _get_dataset_to_type(non_params: Iterable[str]) -> Dict[str, Set[Any]]:
    data_set_to_type: Dict[str, Set[str]] = {}
    for pipe in pipelines.values():
        for node in pipe.nodes:
            sig = inspect.signature(node.func)
            bound_args = (
                sig.bind(**node._inputs)  # noqa: SLF001
                if isinstance(node._inputs, dict)  # noqa: SLF001
                else sig.bind(*node.inputs)
            )
            # get_type_hints does extra work to get the actual type, whereas
            #   inspect.signature returns a string
            try:
                typehints = get_type_hints(
                    node.func.__init__ if inspect.isclass(node.func) else node.func
                )
            except (
                Exception  # pylint: disable=broad-exception-caught # noqa: BLE001
            ) as e:
                logger.warning(
                    "Could not infer type hints for node=%s: %s", node.name, e
                )
                continue
            for param, ds_name in bound_args.arguments.items():
                if (
                    not isinstance(ds_name, str)
                    or ds_name not in non_params
                    or param not in typehints
                ):
                    continue
                data_set_to_type.setdefault(ds_name, set()).add(typehints[param])
    return data_set_to_type


def _create_data_set_entry(
    dataset: str, data_type: Any = ...
) -> Dict[str, Dict[str, str]]:
    """Create a Dataset entry to be written to a kedro catalog.

    Args:
        dataset: Name of the dataset
        data_type: Type hint for the object that `dataset` stores

    Returns:
        Dataset entry
    """
    ds_def = {
        "type": "kedro_mlrun.MLRunDataset",
        # the artifact_id is the same for transcoded versions of the same dataset
        # e.g. df_train@pandas and df_train@parquet read/write the same file
        "artifact_id": dataset.split("@")[0],
    }

    if not isinstance(data_type, type):
        data_type = get_origin(data_type)

    if isinstance(data_type, type):
        model_cls = (
            "MLRunModelDataset" if is_model_dataset_type(data_type) else "MLRunDataset"
        )
        ds_def["type"] = f"kedro_mlrun.{model_cls}"
        ds_def["data_type"] = typehint_to_str(data_type)
    return {dataset: ds_def}


def _format_data_set_entry(
    entry: Dict[str, Dict[str, str]],
    *,
    show_dtype: bool = False,
    indent: int = 2,
) -> str:
    attrs = next(iter(entry.values()))
    stream = StringIO()
    if not show_dtype:
        attrs.pop("data_type", None)
    yaml.dump(entry, stream, indent=indent)
    # add placeholder in case we could not infer the type
    if show_dtype and "data_type" not in attrs:
        stream.write(" " * indent + "# data_type: <type_hint>\n\n")
    return stream.getvalue()


def create_env(metadata: ProjectMetadata, workflow: MLRunWorkflow) -> None:
    """Create a kedro env with its own catalog and parameters files."""
    env_path = metadata.project_path / "conf" / f"{workflow.env}_mlrun"

    # remove old env if exists
    if env_path.exists():
        shutil.rmtree(env_path)

    # create new env
    env_path.mkdir(parents=True)

    session = KedroSession.create(env=workflow.env)
    catalog = session.load_context().catalog

    data_set_to_type = _get_dataset_to_type(non_params=workflow.datasets)

    with open(env_path / "catalog.yml", "w", encoding="utf-8") as catalog_file:
        for dataset in sorted(workflow.datasets):
            dtype = ...
            if dataset in data_set_to_type:
                inferred_type = data_set_to_type[dataset]
                if len(inferred_type) == 1:
                    dtype = next(iter(inferred_type))
                else:
                    logger.warning(
                        (
                            "Multiple type annotations are available for dataset=%s: "
                            "'%s'. "
                            "data_type field will not be set for this catalog entry."
                        ),
                        dataset,
                        ", ".join(typehint_to_str(typ) for typ in inferred_type),
                    )

            entry = _create_data_set_entry(dataset, dtype)
            show_dtype = (
                dataset in workflow.free_inputs
                # we need types for ModelDatasets for model-wrapping hooks to work
                or "MLRunModelDataset" in entry[dataset]["type"]
                # for transcoded datasets, type is used to pick the correct packager
                or "@" in dataset
            )
            entry = _format_data_set_entry(entry, show_dtype=show_dtype)
            catalog_file.write(entry)

    with open(env_path / "parameters.yml", "w", encoding="utf-8") as params_file:
        yaml.dump(catalog.load("parameters"), params_file, indent=2, sort_keys=True)


def add_hook(metadata: ProjectMetadata) -> None:
    settings_fp = metadata.source_dir / metadata.package_name / "settings.py"
    if "add_mlrun_hook" not in settings_fp.read_text(encoding="utf-8"):
        with open(
            str(settings_fp),
            "a",
            encoding="utf-8",
        ) as settings:
            settings.write(
                """
def add_mlrun_hook() -> None:
    from kedro_mlrun import MLRunModelHooks

    _other_hooks = tuple(
        hook
        for hook in globals().get("HOOKS", ())
        if not isinstance(hook, MLRunModelHooks)
    )
    globals()["HOOKS"] = _other_hooks + (MLRunModelHooks(),)


add_mlrun_hook()"""
            )


def create_template(jinja_file: Path) -> jinja2.Template:
    loader = jinja2.FileSystemLoader(jinja_file.parent)
    jinja_env = jinja2.Environment(autoescape=True, loader=loader, lstrip_blocks=True)
    jinja_env.globals.update(
        isinstance=isinstance, str=str, to_project_name=to_project_name
    )
    jinja_env.filters["slugify"] = slugify
    return jinja_env.get_template(jinja_file.name)


def to_project_name(name: str) -> str:
    return re.sub("[^0-9a-zA-Z]+", "-", name).lower()


def to_mlrun_func_name(name: str) -> str:
    """Modifies `name` so that it follows workflow function naming rules."""
    max_mlrun_fn_name = 63
    return re.sub("[^0-9a-zA-Z]+", "-", name).lower().strip("-")[:max_mlrun_fn_name]


def to_variable_name(name: str) -> str:
    """Modifies `name` so that it can be used as a Python variable name."""
    return re.sub("[^_0-9a-zA-Z]+", "_", name).lower().lstrip(string.digits)
