#usr/bin/bash

#PBS -N glxMorph_17Nov

#PBS -k o

#PBS -j oe

#PBS -t 1

#PBS -q long

#PBS -l mem=55gb

module load dot
source /home/myUserName/miniconda3/etc/profile.d/conda.sh
conda activate myenv

(cd /home/myUserName/projects/galaxyMorphology/ && python main.py config/config$PBS_ARRAYID.yaml)
