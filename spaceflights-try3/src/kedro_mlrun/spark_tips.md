# Tips for running Spark jobs with MLRun

You can use Spark jobs within your Workflow, just like any other job.
However, there are some things to keep in mind:

- Spark jobs do not allow
  - fetching source code at execution time
  - passing parameters at execution time
- (Py)Spark version in the job image must be compatible with the Spark version deployed
  in the cluster.
  If the base image is not set, a suitable one will be chosen automatically by MLRun.

`kedro_mlrun` does not support auto-generating a Spark-supporting workflow (yet!).
However, it provides the following templates to help you get started:

- [save_spark.py](./templates/save_spark.py.j2) defines a Spark job and runs it.
- Inside the job, the [kedro_handler_spark.py](./templates/kedro_handler_spark.py.j2)
  script is used to run the Kedro pipeline.

Within your workflow (i.e. `workflow.py`), you can use the Spark job you have created
just like any other function.
Note that even if the rest of the workflow auto-fetches the source code, the Spark job
will not.
