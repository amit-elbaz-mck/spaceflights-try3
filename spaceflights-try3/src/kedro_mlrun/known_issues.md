# Known Issues

## 1. Inconsistent project naming

Users may get "project does not exist" errors or end up with more than one project on
the MLRun/Iguazio server.

### Cause

There are subtle differences between various points where the kedro project name is
sanitised.
Name of the kedro project is sanitised by

- CLI commands, when communicating with the MLRun/Iguazio server, e.g.
  `kedro mlrun log-inputs`
- Templated scripts, which are run by CLI commands, e.g. `save.py`, `mlrun_runner.py`

Ideally, CLI commands and templated scripts must be using the same project name when
communicating with the MLRun server.

Two possible reasons for having inconsistent names are

- CLI commands and the default templated scripts handle the project name differently
  (which is clearly a bug and not intended, please report if you realise that this is
  the case).
- The user has changed the project name in one of the templated scripts but other
  templated scripts or the CLI commands do not have access to this name.

## 2. Empty dataframes cannot be logged

Users can get an error when logging an empty Pandas dataframe.

### Cause

By default, MLRun calculates some summary statistics for dataset artifacts.
However, when the dataframe is empty, MLRun runs into an error when calculate these
statistics.
This is a bug to be fixed on the MLRun side.

A workaround would be to log the artifact as an object but the plugin does not allow
this (yet, see https://github.com/McK-Internal/qblabs-monorepo/issues/267).

## 3. MLRun cannot log/load back my custom object

MLRun provides built-in functionality to log and load back common data types.
However, it is not possible to log/load back all custom objects automatically.

### Solution

Implement a custom [Packager] for your custom object.

Built-in [PandasDataFramePackager][pandas-pkgr] is a good example that can help you get
started.

Then, you can register this packager to your project.

```python
my_proj.add_custom_packager("my_pkg.packager.MyPackager", is_mandatory=False)
```

After registration, MLRun (and hence `kedro-mlrun`) should pick this packager up
automatically and use it to log and load back your custom objects.

> \[!NOTE\]
> The class path you specify `ML.add_custom_packager()` should be importable at
> execution time.
> Make sure you check in the module to your git repo and install any necessary packages
> when building your MLRun Function.

## 4. Cannot install internal packages when building MLRun Function

Packages only available on McKinsey JFrog cannot be installed when building an MLRun
Function.
This is because the image builder pod does not have the necessary pip configuration to
access McKinsey JFrog.

### Workaround

> \[!WARNING\]
> Secrets should be used instead of environment variables in a production setting.
> This workaround uses environment variables and hence should be used with care.

Starting with `mlrun==1.5.0`, it is possible to set environment variables for the
builder pod.

By setting the `PIP_INDEX_URL` and `PIP_EXTRA_INDEX_URL` environment variables, it is
possible to configure pip so that it checks the internal JFrog repository for packages.

Refer to official docs for more information about [configuring pip with environment
variables][pip-env-vars]:

```python
index = "https://JFROG_USERNAME:JFROG_TOKEN@mckinsey.jfrog.io/artifactory/api/pypi/python/simple"


# note that extra_indexes is a single string with multiple urls separated by a space
extra_indexes_ = (
    "https://JFROG_USERNAME:JFROG_TOKEN@mckinsey.jfrog.io/artifactory/api/pypi/EXTRA_REPO_1/simple "
    "https://JFROG_USERNAME:JFROG_TOKEN@mckinsey.jfrog.io/artifactory/api/pypi/EXTRA_REPO_2/simple"
)

project.build_function(
    # ...
    builder_env={"PIP_INDEX_URL": index, "PIP_EXTRA_INDEX_URL": extra_indexes_},
)
# OR
my_func.deploy(
    # ...
    builder_env={"PIP_INDEX_URL": index, "PIP_EXTRA_INDEX_URL": extra_indexes_},
)
```

> \[!WARNING\]
> MLRun passes variables set in `builder_env` as [build arguments][build-args].
> Build arguments are accessible to anyone with access to the image.

## 5. File cannot be found when executing an MLRun Function

When executing an MLRun Function, a python script or module in the codebase cannot be
found.

### Cause

When loading source from a git repo, MLRun copies source code into a directory under
`/tmp`.
As directories under `/tmp` can be randomly deleted by the operating system, the source
code may not be found when the MLRun Function is executed.

### Solution

> \[!NOTE\]
> A fix for this issue is planned to be available with `mlrun>=1.6.0`
> See [this PR](https://github.com/mlrun/mlrun/pull/4461) for more details.

Explicitly set where the source code is copied to by setting the `target_dir` parameter
of `Runtime.with_source_archive()`.

```python
copy_to = "/my_code_dir"
# set target dir explicitly
my_func.with_source_archive(source="git://url-to-my-repo", target_dir=copy_to)
```

## 6. Cannot get logs of a Spark run as pod gets deleted immediately

By default, pods are automatically deleted after termination.

### Solution

Set either or both of the following spark configuration parameters, depending on the
type of pod you are interested in:

```python
my_func.spec.spark_conf["spark.kubernetes.driver.deleteOnTermination"] = False
my_func.spec.spark_conf["spark.kubernetes.executor.deleteOnTermination"] = False
```

## 7. Image build fails

### Possible Cause 1 - Low ephemeral storage

Image builder can fail if it needs more ephemeral storage than what is made available.
This can be seen in the pod details (available on MLRun UI under Projects > _My Proj_ >
Jobs > _My Job_ > Pods or via `kubectl describe pod`).
If this is the case, you should see a message similar to the following:

```text
message: 'The node was low on resource: ephemeral-storage. Threshold quantity:
      27486530764, available: 25891360Ki. Container base was using 7740052Ki, request
      is 0, has larger consumption of ephemeral-storage. '
```

#### Workaround

Try to reduce storage usage within the builder by removing unnecessary files from the
built image.

For example, pip cache can be disabled by adding the following command:

```python
commands = [
    "python -m pip config set global.no-cache-dir false"  # false does disable it
]
my_func.with_commands(commands=commands)
```

Similarly, you can clean apt cache after installing libraries with apt by adding the
following command:

```python
# after installing build-essential, we clean up
commands = [
    # note that the following is a single string formatted nicely for readability
    "sudo apt-get update && "
    "sudo apt-get install -y --no-install-recommends build-essential && "
    "sudo apt-get clean && "
    "sudo rm -rf /var/lib/apt/lists/*"
]
my_func.with_commands(commands=commands)
```

## 8. Spark job fails

### Possible Cause 1 - Working directory not writable

In a Spark job, MLRun adds some JARs to the working directory so that they can be
discovered by Spark.
For this, the working directory must be writable.
However, if the source code for the job is from a git repository, `kaniko` clones the
repo as the user `root`.
Owned by root, this directory is not writable as a non-super-user at execution time.

#### Workaround

Add a `WORKDIR` command to the templated Dockerfile that is used in the image builder
by setting the `extra` parameter of `RuntimeSpec.build`.

For example, the following command sets the working directory to `/igz`, which is
writable at execution time.

```python
spark_job.spec.build.extra = "WORKDIR /igz"
```

[build-args]: https://docs.docker.com/build/guide/build-args/
[packager]: https://docs.mlrun.org/en/stable/api/generated_rsts/mlrun.package.packager.Packager.html#mlrun.package.packager.Packager
[pandas-pkgr]: https://docs.mlrun.org/en/stable/api/generated_rsts/generated_rsts/mlrun.package.packagers.pandas_packagers.PandasDataFramePackager.html#mlrun.package.packagers.pandas_packagers.PandasDataFramePackager
[pip-env-vars]: https://pip.pypa.io/en/stable/topics/configuration/#environment-variables%60%60%60
