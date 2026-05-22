# vLLM installation

To go from a clean state, assume the path where we want to clone `vllm` to be `$HOME/models/modern`. 

We can execute the `setup/clean_vllm.sh` script. It will deactivate the current virtual environment, and remove the repo. You can run it like:

```bash
./setup/clean_vllm.sh $HOME/models/modern
```

Then clone the `vllm` repo specific to your version tag (e.g., v0.21.0) with the parent directory which you should clone the `vllm` repo into. For example, the following below will clone `vllm` into `$HOME/models/modern` and checkout the `v0.21.0` tag.

```bash
./setup/install_vllm.sh $HOME/models/modern v0.21.0
```

The above script will:

- Clone, checkout the tag
- Create a virtual environment
- Install build dependencies
- Build `vllm` and install to the environment
- Install `ray` distributed backend
