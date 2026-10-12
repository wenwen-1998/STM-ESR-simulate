#!/bin/bash
#SBATCH -J input_0.976
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH -p kshcnormal

module purge
source ~/miniconda3/bin/activate

conda activate pyqed


which python
python --version

# 5. 运行你的程序
/usr/bin/time -v python fig2.py

