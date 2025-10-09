#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=xlong
#SBATCH --job-name=no_source_ProcessData
#SBATCH --time=5-11:59:59
#SBATCH --nodes=1              # nodes per instance
#SBATCH --ntasks=1             # tasks per instance
#SBATCH --array=1-1            # instance indexes # CHANGE HOW MANY FILES YOU WANT TO PROCESS
#SBATCH --cpus-per-task=1
#SBATCH --mem=200000

#SBATCH --error=errors/%x-%j.err

#SBATCH --mail-user=mariandelbarrio@gmail.com # CHANGE HERE YOU'RE EMAIL
#SBATCH --mail-type=ALL

echo "JOB TO PROCESS SIGNALS FROM CIGAR"

echo "Slurm job id is ${SLURM_JOB_ID}"
echo "Array job id is ${SLURM_ARRAY_JOB_ID}"
echo "Instance index is ${SLURM_ARRAY_TASK_ID}."


# CHANGE HERE YOUR CIGAR DIRECTORY
CIGAR_DIR="/scratch/marian/CIGAR_ANALYSIS/CIGAR"

cd "${CIGAR_DIR}/scripts";
pwd

# Directory with input .bin files
INPUT_DIR="/data/marian/cigar/ProcessedWaveforms/20250612_no_source_7.5bar_ArXe_9deg" # CHANGE HERE YOUR INPUT DIRECTORY

# Define output file based on input filename (optional customization)
OUTPUT_DIR="../outputs" # HARDCODED (CHECK ProcessData_thr.py SCRIPT)
OUTPUT_DIR="/scratch/marian/cigar/outputs"
OUTPUT_FILE="20250612_ArXe_7.5bar_no_source_9deg_Autotrg_hist"


# Run your Python script
python ProcessData_thr.py ${INPUT_DIR} ${OUTPUT_FILE}

cd ${OUTPUT_DIR}
pwd
ls -lsrth



