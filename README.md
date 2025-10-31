# nccl-debugging

## Useful Readings
* [Networking Chapter from Stas Bekman's Machine Learning Engineering Open Book](https://github.com/stas00/ml-engineering/tree/master/network)
* Communication Benchmarking section in EleutherAI's [Cookbook](https://github.com/EleutherAI/cookbook/tree/main/benchmarks/communication)


## Env setup

    We assume a pixi env, but this should also work with conda/mamba envs.
    See `pixi.toml` for a minimal example env with cuda 12.6 and torch 2.6.0, using the torch supplied nccl version.

    Activate your env.
    ```
    pixi shell
    ```

## Test NCCL inside PyTorch

1. Fill in slurm details in `pytorch-nccl-test.sh`
2. Submit
    ```
    sbatch pytorch-nccl-test.sh
    ```

## Official NVIDIA nccl-tests
In addition to the pytorch test, we can run the official tests from NVIDIA.

1. Clone [NVIDIA nccl-tests repo](https://github.com/NVIDIA/nccl-tests).
    ```
    git clone https://github.com/NVIDIA/nccl-tests.git
    ```


2. Compile `nccl-tests`
    ```bash
    cd nccl-tests
    ../compile_nccl_tests.sh
    ```

3. Run all_reduce performance bench
    See `run_nccl_tests.sh` for an example on how to run nccl-tests with srun.
    
