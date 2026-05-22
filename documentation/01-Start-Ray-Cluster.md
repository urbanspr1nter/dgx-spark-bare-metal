# Starting Ray Cluster

You must SSH individually to each Spark in the cluster and run the `start-ray.sh` script.

If you ever need a fresh copy, you can find them in the project's `cluster` directory where you have these shell scripts:

|Script|DGX Spark|
|------|---------|
|start-ray-spark-a.sh|spark-01|
|start-ray-spark-b.sh|spark-02|
|start-ray-spark-c.sh|spark-03|
|start-ray-spark-d.sh|spark-04|

Copy them and overwrite to: `$HOME/models/start-ray.sh` on each node.