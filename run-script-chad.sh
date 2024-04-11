#!/bin/bash
#SBATCH --job-name="DP-test"     	          # a name for your job
#SBATCH --partition=peregrine-gpu 		  # partition to which job should be submitted
#SBATCH --qos=gpu_short				  # qos type
#SBATCH --nodes=1                		  # node count
#SBATCH --ntasks=1               		  # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        		  # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=16G         		          # total memory per node
#SBATCH --gres=gpu:a100-sxm4-80gb:1               # Request 1 GPU
#SBATCH --time=04:00:00          		  # total run time limit (HH:MM:SS)

source /s/lovelace/c/nobackup/iray/dp-imgclass/SPRING2024/DP-ImageClassificationAuditing/venv/bin/activate
python3 /s/lovelace/c/nobackup/iray/dp-imgclass/SPRING2024/DP-ImageClassificationAuditing/utilities/implementationTrainAudit.py
