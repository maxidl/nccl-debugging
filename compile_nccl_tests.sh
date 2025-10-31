#!/bin/bash
echo "Using env:"$CONDA_PREFIX
python_version=$(python --version | cut -d' ' -f2 | cut -d'.' -f1-2)
echo "Python version: ${python_version}"

export CUDA_HOME=$CONDA_PREFIX
export NCCL_HOME=$CONDA_PREFIX/lib/python${python_version}/site-packages/nvidia/nccl
export MPI_HOME=$CONDA_PREFIX

# create symlinks in the standard lib dir
ln -sf $CONDA_PREFIX/lib/python${python_version}/site-packages/nvidia/nccl/lib/libnccl.so.2 $CONDA_PREFIX/lib/libnccl.so.2
ln -sf $CONDA_PREFIX/lib/python${python_version}/site-packages/nvidia/nccl/lib/libnccl.so.2 $CONDA_PREFIX/lib/libnccl.so

# make sure the paths are visible to the compiler
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python${python_version}/site-packages/nvidia/nccl/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CONDA_PREFIX/lib/python${python_version}/site-packages/nvidia/nccl/lib:$CONDA_PREFIX/lib:$LIBRARY_PATH
export CPATH=$CONDA_PREFIX/lib/python${python_version}/site-packages/nvidia/nccl/include:$CPATH
ls -la $CONDA_PREFIX/lib/libnccl*

echo "--------------------------------"
echo "Compiling nccl-tests with MPI=1"
make clean
make MPI=1
if [ $? -eq 0 ]; then
echo "nccl-tests compilation succeeded"
else
echo "nccl-tests compilation failed"
exit 1
fi
