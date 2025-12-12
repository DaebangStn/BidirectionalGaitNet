#!/bin/bash
# Add a patient directory with standard structure
# Usage: add_patient.sh <pid>

BASE_DIR="/mnt/blue8T/CP/RM"

if [ -z "$1" ]; then
    echo "Usage: $0 <pid>"
    exit 1
fi

PID="$1"
PATIENT_DIR="$BASE_DIR/$PID"

if [ -d "$PATIENT_DIR" ]; then
    echo "  [EXISTS] $PATIENT_DIR"
else
    # Create directory structure
    mkdir -p "$PATIENT_DIR/gait"
    mkdir -p "$PATIENT_DIR/anatomy/skeleton"
    mkdir -p "$PATIENT_DIR/anatomy/muscle"
    mkdir -p "$PATIENT_DIR/ckpt"
    mkdir -p "$PATIENT_DIR/dicom"

    # Create empty metadata.yaml
    touch "$PATIENT_DIR/metadata.yaml"
    touch "$PATIENT_DIR/rom.yaml"

    echo "  [CREATED] $PATIENT_DIR"
fi
