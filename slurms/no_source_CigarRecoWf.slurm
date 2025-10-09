#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=no_source_RecoWvf
#SBATCH --time=0-23:59:59
#SBATCH --nodes=1              # nodes per instance
#SBATCH --ntasks=1             # tasks per instance
#SBATCH --array=1-26           # instance indexes # CHANGE HOW MANY FILES YOU WANT TO PROCESS
#SBATCH --cpus-per-task=1
#SBATCH --mem=200000

#SBATCH --error=errors/%x-%A_%a.err

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
INPUT_DIR="/data/marian/cigar/runs/20250620_ArXe_3.5bar_no_source_6deg_Autotrg" # CHANGE HERE YOUR INPUT DIRECTORY

# Get sorted list of .bin files and select one using the array index
INPUT_FILE=$(ls ${INPUT_DIR}/*.bin | sort | sed -n "$((SLURM_ARRAY_TASK_ID))p")

# Define output file based on input filename (optional customization)
OUTPUT_DIR="${CIGAR_DIR}/outputs"
OUTPUT_FILE="${OUTPUT_DIR}/20250620_ArXe_3.5bar_no_source_6deg_Autotrg_output_${SLURM_ARRAY_TASK_ID}.csv" # CHANGE HERE YOUR OUTPUT FILE


# Run your Python script
python RecoWf_V1.py -I ${INPUT_FILE} -O ${OUTPUT_FILE} -C params/reco_param.txt

cd ${OUTPUT_DIR}
pwd
ls -lsrth



