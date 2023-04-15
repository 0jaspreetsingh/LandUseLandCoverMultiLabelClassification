#!/bin/bash

#SBATCH --nodes=1               # Number of nodes or servers. See: http://koeln.kl.dfki.de:3000/d/slurm-resources/resources?orgId=1&refresh=15s
#SBATCH --ntasks-per-node=1     # Number of task in each node, we want 1 
#SBATCH --cpus-per-task=4       # We want 4 cores for this job.
#SBATCH --mem-per-cpu=16gb      # each core to have 16 Gb RAM
#SBATCH --gres=gpu:0            # We want 0 GPUs in each node for this job.
#SBATCH --partition=RTXA6000,V100-32GB,RTX3090  # Run this only in these mentioned GPUs. If you don't have any choice over GPUs, remove this parameter.
#SBATCH --job-name=create_datadings

NOW=$( date '+%F-%H-%M-%S' )
JOB_NAME=create_datadings

srun -K\
  --container-mounts="`pwd`":"`pwd`",/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro \
  --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_22.11-py3.sqsh \
  --container-workdir="`pwd`" \
  --task-prolog=/home/jsingh/Projects/LandCoverClassifier/multispectral-lulc/install.sh \
  python -m create_datadings --src-dir "/ds/images/BigEarthNet/BigEarthNet-S2/BigEarthNet-v1.0/" --dest-dir "/netscratch/$USER/float32_dataset/"