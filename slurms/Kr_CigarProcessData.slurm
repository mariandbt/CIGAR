#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=xlong
#SBATCH --job-name=Kr_ProcessData
#SBATCH --time=5-11:59:59
#SBATCH --nodes=1              # nodes per instance
#SBATCH --ntasks=1             # tasks per instance
#SBATCH --array=1-1            # instance indexes
#SBATCH --cpus-per-task=1
#SBATCH --mem=200000

#SBATCH --error=errors/%x-%j.err

#SBATCH --mail-user=mariandelbarrio@gmail.com
#SBATCH --mail-type=ALL

echo "JOB TO PROCESS SIGNALS FROM CIGAR"

echo "Slurm job id is ${SLURM_JOB_ID}"
echo "Array job id is ${SLURM_ARRAY_JOB_ID}"
echo "Instance index is ${SLURM_ARRAY_TASK_ID}."

cd /data/marian/RecoWf;
pwd

# Directory with input .bin files
INPUT_DIR="/data/marian/cigar/ProcessedWaveforms/20250612_Kr_7.5bar_ArXe_9deg"

# Define output file based on input filename (optional customization)
OUTPUT_DIR="/scratch/marian/cigar/outputs"
OUTPUT_FILE="20250612_ArXe_7.5bar_Kr_9deg_Autotrg_hist"


# Run your Python script
python ProcessData_thr.py ${INPUT_DIR} ${OUTPUT_FILE}

cd ${OUTPUT_DIR}
pwd
ls -lsrth
# mv ${OUTPUT_DIR}/${OUTPUT_FILE}  /data/marian/cigar/outputs



