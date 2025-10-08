#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=no_source_RecoWvf
#SBATCH --time=0-23:59:59
#SBATCH --nodes=1              # nodes per instance
#SBATCH --ntasks=1             # tasks per instance
#SBATCH --array=1-1            # instance indexes
#SBATCH --cpus-per-task=1
#SBATCH --mem=200000

#SBATCH --error=errors/%x-%A_%a.err

#SBATCH --mail-user=mariandelbarrio@gmail.com
#SBATCH --mail-type=ALL

echo "JOB TO PROCESS SIGNALS FROM CIGAR"

echo "Slurm job id is ${SLURM_JOB_ID}"
echo "Array job id is ${SLURM_ARRAY_JOB_ID}"
echo "Instance index is ${SLURM_ARRAY_TASK_ID}."

cd /data/marian/RecoWf;
pwd

ic_activate

# Directory with input .bin files
INPUT_DIR="/data/marian/cigar/runs/20250818_Air_no_source_10deg_54.35V_Autotrg_DCR"

# Get sorted list of .bin files and select one using the array index
INPUT_FILE=$(ls ${INPUT_DIR}/*.bin | sort | sed -n "$((SLURM_ARRAY_TASK_ID))p")

# Define output file based on input filename (optional customization)
OUTPUT_DIR="/scratch/marian/cigar/outputs"
OUTPUT_FILE="${OUTPUT_DIR}/20250818_Air_no_source_10deg_54.35V_Autotrg_DCR_output_${SLURM_ARRAY_TASK_ID}.csv"


# Run your Python script
python RecoWf_V1.py -I ${INPUT_FILE} -O ${OUTPUT_FILE} -C /data/marian/RecoWf/reco_param.txt

cd ${OUTPUT_DIR}
pwd
ls -lsrth
# mv ${OUTPUT_FILE}  /data/marian/cigar/outputs



