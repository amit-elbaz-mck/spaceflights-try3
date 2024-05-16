# `kedro-mlrun`

## Overview

`kedro-mlrun` is a Kedro plugin designed to facilitate the seamless
deployment of your Kedro pipelines on MLRun and the Iguazio platform.
Here's an overview of what `kedro-mlrun` offers:

1. **`kedro-mlrun` Command Group**: `kedro-mlrun` extends the Kedro CLI by introducing
   the `kedro mlrun` command group.
   This command group provides a set of CLI commands specifically designed for
   deployment to MLRun.
   It simplifies the deployment workflow and allows you to interact with the deployment
   process using familiar Kedro commands.
2. **Kedro hooks for target-specific capabilities**: `kedro-mlrun` incorporates Kedro
   hooks, which are functions that can be integrated into the Kedro pipeline to enable
   MLRun-specific capabilities during deployment.
   These hooks allow you to customize your pipeline for MLrun, ensuring smooth
   execution and efficient utilization of platform resources.
3. **Kedro Dataset implementations for target filesystems**: `kedro-mlrun` provides
   specialized Kedro Dataset implementations that enable seamless reading from and
   writing to MLRun filesystem.
   These implementations are tailored to the requirements and specifications of MLRun,
   ensuring compatibility and optimal data handling during deployment.

With `kedro-mlrun`, you can easily deploy your Kedro pipelines on MLRun and take
advantage of MLRun-specific capabilities paired with streamlined deployment experience.

## Installation

`kedro-mlrun` is available on JFrog Artifactory. To install, run:

```shell
pip install mck.kedro-mlrun
```

## Usage

Once inside a Kedro project directory, you can see that an additional command group,
`mlrun`, is available to you:

```shell
$ cd PATH/TO/MY/KEDRO_PROJECT
$ kedro mlrun --help

Usage: kedro mlrun [OPTIONS] COMMAND [ARGS]...

  Kedro mlrun.

Options:
  -h, --help  Show this message and exit.

Commands:
  config      Set config value for MLRun.
  init        Initialize MLRun features for this kedro project.
  log-inputs  Log global free inputs.
  rebuild     Rebuild MLRun files from kedro project.
  run         Run project on remote Iguazio environment.
  save        Push project to remote environment.
```

Check out the [Tutorial: Deploy Your Spaceflights](./tutorial.md) to get started!

There is also a [Spark tips](./spark_tips.md) page with some tips for using Spark in
your deployment.

## Known Issues

See [here](./known_issues.md).
