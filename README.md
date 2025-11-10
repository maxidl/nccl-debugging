# nccl-debugging

## Useful Readings
* [Networking Chapter from Stas Bekman's Machine Learning Engineering Open Book](https://github.com/stas00/ml-engineering/tree/master/network)
* Communication Benchmarking section in EleutherAI's [Cookbook](https://github.com/EleutherAI/cookbook/tree/main/benchmarks/communication)


## Env setup

We assume a pixi env, but this should also work with conda/mamba envs.
See `pixi.toml` for example envs with different cuda + torch versions, using the torch supplied nccl version.

Activate your env.
```bash
pixi shell -e torch290cu128 # example for torch 2.9.0 with cuda 12.8
```

If you have to activate the env within the job, you can add the following at the top of sbatch scripts:

```bash
# activate pixi env in current shell
eval "$(pixi shell-hook -e torch290cu128)" # example for torch 2.9.0 with cuda 12.8
echo "python: $(which python)"
```

## Test NCCL inside PyTorch

1. Fill in slurm details in `pytorch-nccl-test.sh`
2. Submit
    ```bash
    sbatch pytorch-nccl-test.sh
    ```

## Official NVIDIA nccl-tests
In addition to the pytorch test, we can run the official tests from NVIDIA.

1. Clone [NVIDIA nccl-tests repo](https://github.com/NVIDIA/nccl-tests).
    ```bash
    git clone https://github.com/NVIDIA/nccl-tests.git
    ```


2. Compile `nccl-tests`
    ```bash
    cd nccl-tests
    ../compile_nccl_tests.sh
    ```

3. Run all_reduce performance bench
    See `run_nccl_tests.sh` for an example on how to run nccl-tests with srun.
    
