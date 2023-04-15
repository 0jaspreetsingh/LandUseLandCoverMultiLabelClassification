#!/bin/bash

#SBATCH --nodes=1               # Number of nodes or servers. See: http://koeln.kl.dfki.de:3000/d/slurm-resources/resources?orgId=1&refresh=15s
#SBATCH --ntasks-per-node=1     # Number of task in each node, we want 1 
#SBATCH --cpus-per-task=4       # We want 4 cores for this job.
#SBATCH --mem-per-cpu=24gb      # each core to have 16 Gb RAM
#SBATCH --gres=gpu:1            # We want 1 GPUs in each node for this job.
#SBATCH --partition=RTXA6000,V100-32GB,RTX3090  # Run this only in these mentioned GPUs. If you don't have any choice over GPUs, remove this parameter.
#SBATCH --job-name=ResnetIntermediateFusion
#SBATCH -e ./out/R-%x.%j.out
#SBATCH -o ./out/R-%x.%j.out
#SBATCH --open-mode=append

NOW=$( date '+%F-%H-%M-%S' )
JOB_NAME=ResnetIntermediateFusion

srun -K\
  --container-mounts="`pwd`":"`pwd`",/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro, \
  --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_22.11-py3.sqsh \
  --container-workdir="`pwd`" \
  --task-prolog=/home/jsingh/Projects/jayanth_jaspreet-dev-iter2/scripts/install.sh \
  python -m main --root-dir "/netscratch/$USER/float32_dataset" --num-workers 1 --batch-size 64 --out-dir "/netscratch/$USER/dataset" --model-choice "resnet_intermediate" --train-countries "all_bands" --test-countries "all_bands" --notes "training all bands with resnetIntermediateFusion"
