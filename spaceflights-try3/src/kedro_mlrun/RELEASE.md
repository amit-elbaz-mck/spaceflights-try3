# Release History

## Upcoming

- Safer type inference
  ([#333](https://github.com/McK-Internal/qblabs-monorepo/pull/333))
- Validate that local file exists before logging the artifact
  ([#121](https://github.com/McK-Internal/qblabs-monorepo/pull/121))
- Fix wrong artifact id for transcoded datasets
  ([#368](https://github.com/McK-Internal/qblabs-monorepo/pull/368))
- Add template files and documentation for Spark job
  ([#370](https://github.com/McK-Internal/qblabs-monorepo/pull/370))
- Fix errors identified by ruff
  ([#308](https://github.com/McK-Internal/qblabs-monorepo/pull/308))

## v0.0.7

- It is now possible to pass kwargs for packing and logging, separately.
  ([#310](https://github.com/McK-Internal/qblabs-monorepo/pull/310))
- Custom packagers of the project are loaded before unpacking in `MLRunDataset`
  ([#310](https://github.com/McK-Internal/qblabs-monorepo/pull/310))

## v0.0.6

- Better handling of git-related errors in `save.py`
  ([#249](https://github.com/McK-Internal/qblabs-monorepo/pull/249))

## v0.0.5

- Update package data declaration
  ([#244](https://github.com/McK-Internal/qblabs-monorepo/pull/244))

## v0.0.4

- Add "deploying kedro" document
  ([#189](https://github.com/McK-Internal/qblabs-monorepo/pull/189))

## v0.0.3

- Add example script under `examples/`
  ([#216](https://github.com/McK-Internal/qblabs-monorepo/pull/216))

## v0.0.2

- Rename package to `kedro-mlrun`
  ([#202](https://github.com/McK-Internal/qblabs-monorepo/pull/202))
- Drop support for `mlrun<1.4` and `python<3.9`
  ([#182](https://github.com/McK-Internal/qblabs-monorepo/pull/182))
- Fix repo url parsing for "https://" scheme in the `save.py` template
  ([#183](https://github.com/McK-Internal/qblabs-monorepo/pull/183))
- Multiple fixes in the `workflow.py` template
  ([#184](https://github.com/McK-Internal/qblabs-monorepo/pull/184),
  [#185](https://github.com/McK-Internal/qblabs-monorepo/pull/185),
  [#196](https://github.com/McK-Internal/qblabs-monorepo/pull/196))
- Rename 'DataSet' to 'Dataset' ([#212](https://github.com/McK-Internal/qblabs-monorepo/pull/212))

## v0.0.1

- Initial release
