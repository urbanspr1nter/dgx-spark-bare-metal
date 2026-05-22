# Infrastructure

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

# DGX Spark IP Addresses

Use IPv4 to access the DGX Sparks. This network is not configured to access by hostname.

|hostname|IP|
|--------|--|
|spark-01|192.168.1.21|
|spark-02|192.168.1.40|
|spark-03|192.168.1.24|
|spark-04|192.168.1.48|