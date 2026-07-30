conda activate /home/sci/work/zhw_envs/dino_worker

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH

export TORCH_LIB=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")

export LD_LIBRARY_PATH=$TORCH_LIB:$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$CONDA_PREFIX/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

for d in $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/lib; do
  export LD_LIBRARY_PATH=$d:$LD_LIBRARY_PATH
done