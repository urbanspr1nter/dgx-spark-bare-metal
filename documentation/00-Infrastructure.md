# Infrastructure

**CRITICAL** - All DGX Sparks in the network are already enabled for RDMA access. They can talk to each other already. Don't over-index on that.

The 4x DGX Spark cluster setup allows us to have effectively 480 GB (4x120 GB) of usable memory for inference.

Note: We don't use all of the specified 128 GB each DGX Spark has since we need to reserve some memory for the operating system.

The user name for each DGX Spark and all other computers (including this machine) in this internal network can be figured through running the command:

```bash
echo $USER
```

For example, if you want to SSH:

```bash
ssh "$USER@192.168.1.21"
```

# File System Layout

**CRITICAL** - Each DGX Spark node has the _same_ filesystem layout. 

For example, this means this path is consistent for all nodes:

```sh
$HOME/models/start-ray.sh
```

Or 

```
$HOME/models/modern/vllm
```

This means you should be able to just `ssh` into each node and expect to find the same paths and common files.

# Development Environment

**CRITICAL** The main DGX Spark is `spark-01`. Here you have more tools available than the other Sparks. 

Most importantly you live here!

- You have passwordless SSH to all Sparks.
- You have passwordless `sudo` in all Sparks.
- You have Python, CUDA and Node installed properly.
- You are running under `pi` coding agent.

# DGX Spark IP Addresses

Use IPv4 to access the DGX Sparks. This network is not configured to access by hostname.

|hostname|IP|
|--------|--|
|spark-01|192.168.1.21|
|spark-02|192.168.1.40|
|spark-03|192.168.1.24|
|spark-04|192.168.1.48|

# Network Info

To find the interface details you can run:

`ibdev2netdev`

Example output:

```
rocep1s0f0 port 1 ==> enp1s0f0np0 (Up)
rocep1s0f1 port 1 ==> enp1s0f1np1 (Down)
roceP2p1s0f0 port 1 ==> enP2p1s0f0np0 (Up)
roceP2p1s0f1 port 1 ==> enP2p1s0f1np1 (Down)
```

Ideally, this should rarely ever change. I don't mess with physical things that often.