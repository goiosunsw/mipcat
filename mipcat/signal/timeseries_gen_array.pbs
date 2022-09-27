#!/bin/bash

#PBS -l select=1:ncpus=4:mem=8gb
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -J 1-30

codedir=${HOME}/Devel/sensor_clarinet_processing
scriptdir=$codedir/collection
configdir=$codedir/runsheets
outdir=/srv/scratch/z3227932/tmp_results

eval "$(/home/z3227932/miniconda3/bin/conda shell.bash hook)"
conda activate sensor_clarinet

cd ${PBS_O_WORKDIR}

# Uncomment for testing in interactive
#PBS_ARRAY_INDEX=1

python $scriptdir/file_ts_gen.py  $configdir/combined_config.yaml ${PBS_ARRAY_INDEX} ${TMPDIR}
rsync -r ${TMPDIR}/ ${outdir}
