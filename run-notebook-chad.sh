#!/bin/bash
#SBATCH --job-name="DP-test"                   # a name for your job
#SBATCH --partition=kestrel-gpu                   # partition to which job should be submitted
#SBATCH --qos=gpu_short                           # qos type
#SBATCH --nodes=1                                 # node count
#SBATCH --ntasks=1                                # total number of tasks across all nodes
#SBATCH --cpus-per-task=1                         # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=16G                                 # total memory per node
#SBATCH --gres=gpu:3090:1                         # Request 1 GPU
#SBATCH --time=06:00:00                           # total run time limit (HH:MM:SS)

source /s/lovelace/c/nobackup/iray/dp-imgclass/private_vision_clone_1/venv/bin/activate

port=15247
ssh -N -f -R $port:localhost:$port falcon

jupyter-notebook --no-browser --port=$port
