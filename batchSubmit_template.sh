#usr/bin/bash

#PBS -N nameOfTheJob

#PBS -k o

#PBS -j oe

#PBS -t 3

module load dot
source /home/myUserName/miniconda3/etc/profile.d/conda.sh
conda activate myenv

(cd /home/myUserName/projects/galaxyMorphology/ && python main.py config/config$PBS_ARRAYID.yaml)
