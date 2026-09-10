#!/bin/bash

# CHANGE HERE YOUR CIGAR DIRECTORY
CIGAR_DIR="/home/marian/CIGAR_ANALYSIS/CIGAR"

RUN="20251016_Xe_no_source_1.5bar_17deg_54.5V_Majority"


cd "${CIGAR_DIR}/scripts";
pwd

# Directory with input .bin files
#INPUT_DIR="/data/marian/cigar/ProcessedWaveforms/20250612_no_source_7.5bar_ArXe_9deg" # CHANGE HERE YOUR INPUT DIRECTORY
INPUT_DIR="../outputs" # CHANGE HERE YOUR INPUT DIRECTORY

# Define output file based on input filename (optional customization)
OUTPUT_DIR="../outputs" # HARDCODED (CHECK ProcessData_thr.py SCRIPT)
OUTPUT_FILE="${RUN}_hist"


# Run your Python script
python ProcessData_thr.py ${INPUT_DIR} ${OUTPUT_FILE}

cd ${OUTPUT_DIR}
pwd
ls -lsrth



