export CONDA_BASE="$HOME/miniconda3"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate xsuite-py310

srun --partition=gpu_h100_interactive --qos=gpu --gres=gpu:h100:1 --cpus-per-task=4 --mem=16G --time=02:00:00 --pty bash

jupyter notebook --no-browser --port=8889