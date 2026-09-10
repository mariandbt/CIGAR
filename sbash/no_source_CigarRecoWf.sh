#!/bin/bash

# CHANGE HERE YOUR CIGAR DIRECTORY
CIGAR_DIR="/home/marian/CIGAR_ANALYSIS/CIGAR"

RUN="20251016_Xe_no_source_1.5bar_17deg_54.5V_Majority"

# Loop over a sequence of task IDs
for TASK_ID in {5..14}; do   # change 1..10 to however many .bin files you have

    cd "${CIGAR_DIR}/scripts";
    pwd

    # Directory with input .bin files
    INPUT_DIR="/home/marian/CIGAR_ANALYSIS/CIGAR/data/${RUN}" # CHANGE HERE YOUR INPUT DIRECTORY

    # Get sorted list of .bin files and select one using the array index
    INPUT_FILE=$(ls ${INPUT_DIR}/*.bin | sort | sed -n "$((${TASK_ID}))p")

    # Define output file based on input filename (optional customization)
    OUTPUT_DIR="${CIGAR_DIR}/outputs"
    OUTPUT_FILE="${OUTPUT_DIR}/${RUN}_output_${TASK_ID}.csv" # CHANGE HERE YOUR OUTPUT FILE


    # Run your Python script
    python RecoWf_V1.py -I ${INPUT_FILE} -O ${OUTPUT_FILE} -C params/reco_param.txt

    cd ${OUTPUT_DIR}
    pwd
    ls -lsrth

done



