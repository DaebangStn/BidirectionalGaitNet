#!/usr/bin/env python3
"""
Physical Examination CSV to YAML Exporter

Parses data/PhysicalExamination.csv and exports to YAML with descriptive comments.
"""

import csv
import re
from pathlib import Path
from datetime import date
from typing import Any, Optional

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq


# Column indices (0-indexed)
COL_PATIENT_ID = 0
COL_NAME = 1
COL_SER_NO = 2
COL_GMFCS = 3
COL_PRE_OP_AGE = 4
COL_PRE_OP_ROM_START = 5  # 35 fields (5-39) including side
COL_POST_OP_AGE = 40
COL_POST_OP_ROM_START = 41  # 35 fields (41-75)
COL_SURGERY1_AGE = 76
COL_SURGERY1_NAME = 77
COL_SURGERY2_AGE = 78
COL_SURGERY2_NAME = 79
COL_SURGERY1_CT_AGE = 80
COL_SURGERY1_CT_NAME = 81
COL_SURGERY1_XRAY1_AGE = 82
COL_SURGERY1_XRAY1_NAME = 83
COL_SURGERY1_XRAY2_AGE = 84
COL_SURGERY1_XRAY2_NAME = 85
COL_SURGERY2_CT_AGE = 86
COL_SURGERY2_CT_NAME = 87
COL_SURGERY2_XRAY1_AGE = 88
COL_SURGERY2_XRAY1_NAME = 89
COL_SURGERY2_XRAY2_AGE = 90
COL_SURGERY2_XRAY2_NAME = 91
COL_SURGERY2_XRAY3_AGE = 92
COL_SURGERY2_XRAY3_NAME = 93
COL_SURGERY2_XRAY4_AGE = 94
COL_SURGERY2_XRAY4_NAME = 95

# ROM field indices within the 32-field block (relative offset)
ROM_FIELDS = [
    ("side", "int"),  # 0: 0=Right, 1=Left
    ("hip.further_flexion", "float"),  # 1
    ("hip.extension_thomas", "float"),  # 2: Thomas test = FC
    ("hip.extension_staheli", "float"),  # 3: Staheli test
    ("hip.abduction_ext_r1", "float"),  # 4: R1 = fast
    ("hip.abduction_ext_r2", "float"),  # 5
    ("hip.abduction_flex90_r1", "float"),  # 6
    ("hip.abduction_flex90_r2", "float"),  # 7
    ("hip.adduction_r1", "float"),  # 8
    ("hip.adduction_r2", "float"),  # 9
    ("hip.external_rotation", "float"),  # 10: Prone
    ("hip.internal_rotation", "float"),  # 11: Prone
    ("hip.anteversion", "float"),  # 12: Prone
    ("ankle.thigh_foot_angle", "float"),  # 13: TMA
    ("knee.flexion_contracture", "float"),  # 14
    ("knee.further_flexion", "float"),  # 15
    ("knee.popliteal_unilateral", "float"),  # 16
    ("knee.popliteal_bilateral", "float"),  # 17
    ("knee.duncan_ely", "duncan_ely"),  # 18: +/- or numeric
    ("ankle.plantarflexion", "float"),  # 19
    ("ankle.dorsiflexion_knee90_r1", "float"),  # 20
    ("ankle.dorsiflexion_knee90_r2", "float"),  # 21
    ("ankle.dorsiflexion_knee0_r1", "float"),  # 22
    ("ankle.dorsiflexion_knee0_r2", "float"),  # 23
    ("foot.hindfoot_valgus", "bool"),  # 24: Y/N
    ("foot.hindfoot_varus", "bool"),  # 25: Y/N
    ("foot.hindfoot_flexibility", "bool"),  # 26: Y/N
    ("foot.midfoot_tn_pronation", "bool"),  # 27: Y/N
    ("foot.midfoot_tn_supination", "bool"),  # 28: Y/N
    ("foot.midfoot_flexibility", "bool"),  # 29: Y/N
    ("foot.forefoot_adduction", "bool"),  # 30: Y/N
    ("foot.forefoot_abduction", "bool"),  # 31: Y/N
    ("foot.forefoot_flexibility", "bool"),  # 32: Y/N
    ("muscle_tone.quadriceps", "roman"),  # 33: Roman numeral
    ("muscle_tone.eph", "roman"),  # 34: Roman numeral
]

# Roman numeral mapping
ROMAN_TO_INT = {
    "Ⅰ": 1, "Ⅱ": 2, "Ⅲ": 3, "Ⅳ": 4, "Ⅴ": 5,
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5,
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
}


def parse_value(val: str, dtype: str) -> Optional[Any]:
    """Convert string value to appropriate type."""
    val = val.strip()

    if not val or val == "-":
        return None

    if dtype == "int":
        try:
            return int(float(val))
        except ValueError:
            return None

    elif dtype == "float":
        try:
            return float(val)
        except ValueError:
            return None

    elif dtype == "bool":
        return val.upper() == "Y"

    elif dtype == "duncan_ely":
        # "+" or numeric → True, "-" → False
        if val == "-":
            return False
        elif val == "+":
            return True
        else:
            # Try to parse as number - if it's a number, it's positive
            try:
                float(val)
                return True
            except ValueError:
                return None

    elif dtype == "roman":
        # Handle variants like "Ⅲ+", "Ⅴ-", "0"
        val_clean = val.rstrip("+-").strip()
        if val_clean == "0":
            return 0
        for roman, num in ROMAN_TO_INT.items():
            if roman in val_clean:
                return num
        return None

    elif dtype == "str":
        return val if val else None

    return val


def parse_rom_fields(row: list, start_col: int) -> dict:
    """Parse 35 ROM fields starting at start_col into nested structure."""
    rom = {
        "hip": {},
        "knee": {},
        "ankle": {},
        "foot": {},
        "muscle_tone": {},
    }

    for i, (field_path, dtype) in enumerate(ROM_FIELDS):
        if i == 0:  # Skip side field
            continue

        col_idx = start_col + i
        if col_idx >= len(row):
            continue

        val = parse_value(row[col_idx], dtype)

        # Parse field path like "hip.further_flexion"
        parts = field_path.split(".")
        if len(parts) == 2:
            category, field = parts
            if category in rom:
                rom[category][field] = val

    return rom


def parse_surgery_imaging(row: list, surgery_num: int) -> dict:
    """Parse CT and X-ray data for a surgery."""
    if surgery_num == 1:
        ct_age_col = COL_SURGERY1_CT_AGE
        ct_name_col = COL_SURGERY1_CT_NAME
        xray_cols = [
            (COL_SURGERY1_XRAY1_AGE, COL_SURGERY1_XRAY1_NAME),
            (COL_SURGERY1_XRAY2_AGE, COL_SURGERY1_XRAY2_NAME),
        ]
    else:
        ct_age_col = COL_SURGERY2_CT_AGE
        ct_name_col = COL_SURGERY2_CT_NAME
        xray_cols = [
            (COL_SURGERY2_XRAY1_AGE, COL_SURGERY2_XRAY1_NAME),
            (COL_SURGERY2_XRAY2_AGE, COL_SURGERY2_XRAY2_NAME),
            (COL_SURGERY2_XRAY3_AGE, COL_SURGERY2_XRAY3_NAME),
            (COL_SURGERY2_XRAY4_AGE, COL_SURGERY2_XRAY4_NAME),
        ]

    ct = {
        "age": parse_value(row[ct_age_col] if ct_age_col < len(row) else "", "float"),
        "name": parse_value(row[ct_name_col] if ct_name_col < len(row) else "", "str"),
    }

    xrays = []
    for age_col, name_col in xray_cols:
        if age_col < len(row) and name_col < len(row):
            age = parse_value(row[age_col], "float")
            name = parse_value(row[name_col], "str")
            if age is not None or name is not None:
                xrays.append({"age": age, "name": name})

    return {"ct": ct, "xray": xrays}


def parse_physical_examination(csv_path: str) -> dict:
    """Parse PhysicalExamination.csv into structured dict."""
    patients = {}

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Find first data row (has numeric patient ID in column 0)
    data_start = 0
    for i, row in enumerate(rows):
        col0 = row[0].strip() if row else ''
        if col0.isdigit() and len(col0) >= 7:
            data_start = i
            break

    data_rows = rows[data_start:]

    # Process rows in pairs
    i = 0
    while i < len(data_rows):
        row1 = data_rows[i]

        # Skip empty rows
        if not row1 or not any(cell.strip() for cell in row1[:5]):
            i += 1
            continue

        # Get patient info from first row
        patient_id = row1[COL_PATIENT_ID].strip()

        # Skip if no patient_id (might be a continuation row)
        if not patient_id or not patient_id.isdigit():
            i += 1
            continue

        name = row1[COL_NAME].strip()
        ser_no = parse_value(row1[COL_SER_NO], "int")
        gmfcs = parse_value(row1[COL_GMFCS], "int")

        # Get second row (left side data)
        row2 = data_rows[i + 1] if i + 1 < len(data_rows) else []

        # Determine which is right and which is left based on side field
        side1 = parse_value(row1[COL_PRE_OP_ROM_START], "int")
        side2 = parse_value(row2[COL_PRE_OP_ROM_START], "int") if row2 else None

        # Parse ROM data
        if side1 == 0:  # First row is right
            right_row, left_row = row1, row2
        else:  # First row is left (unlikely but handle)
            right_row, left_row = row2, row1

        pre_op_rom = {
            "right": parse_rom_fields(right_row, COL_PRE_OP_ROM_START) if right_row else {},
            "left": parse_rom_fields(left_row, COL_PRE_OP_ROM_START) if left_row else {},
        }

        post_op_rom = {
            "right": parse_rom_fields(right_row, COL_POST_OP_ROM_START) if right_row else {},
            "left": parse_rom_fields(left_row, COL_POST_OP_ROM_START) if left_row else {},
        }

        # Parse surgery info (from first row only)
        surgery1_age = parse_value(row1[COL_SURGERY1_AGE] if COL_SURGERY1_AGE < len(row1) else "", "float")
        surgery1_name = parse_value(row1[COL_SURGERY1_NAME] if COL_SURGERY1_NAME < len(row1) else "", "str")
        surgery2_age = parse_value(row1[COL_SURGERY2_AGE] if COL_SURGERY2_AGE < len(row1) else "", "float")
        surgery2_name = parse_value(row1[COL_SURGERY2_NAME] if COL_SURGERY2_NAME < len(row1) else "", "str")

        surgery1_imaging = parse_surgery_imaging(row1, 1)
        surgery2_imaging = parse_surgery_imaging(row1, 2)

        patients[patient_id] = {
            "name": name,
            "ser_no": ser_no,
            "gmfcs": gmfcs,
            "pre_op": {
                "age": parse_value(row1[COL_PRE_OP_AGE], "float"),
                "rom": pre_op_rom,
            },
            "post_op": {
                "age": parse_value(row1[COL_POST_OP_AGE], "float"),
                "rom": post_op_rom,
            },
            "surgery1": {
                "age": surgery1_age,
                "name": surgery1_name,
                **surgery1_imaging,
            },
            "surgery2": {
                "age": surgery2_age,
                "name": surgery2_name,
                **surgery2_imaging,
            },
        }

        # Move to next patient (skip 2 rows)
        i += 2

    return patients


def to_commented_map(data: dict) -> CommentedMap:
    """Convert dict to CommentedMap recursively."""
    cm = CommentedMap()
    for k, v in data.items():
        if isinstance(v, dict):
            cm[k] = to_commented_map(v)
        elif isinstance(v, list):
            cm[k] = CommentedSeq([to_commented_map(item) if isinstance(item, dict) else item for item in v])
        else:
            cm[k] = v
    return cm


def add_yaml_comments(yaml_data: CommentedMap, patients_data: dict):
    """Add descriptive comments to YAML structure."""

    # Add comments to each patient
    for patient_id, patient in yaml_data["patients"].items():
        name = patients_data[patient_id].get("name", "")
        yaml_data["patients"].yaml_add_eol_comment(f"{name}", patient_id)

        # GMFCS comment
        if "gmfcs" in patient:
            patient.yaml_add_eol_comment("Gross Motor Function Classification System (1-5)", "gmfcs")

        # Pre-op comments
        if "pre_op" in patient:
            patient["pre_op"].yaml_add_eol_comment("years", "age")
            if "rom" in patient["pre_op"]:
                for side in ["right", "left"]:
                    if side in patient["pre_op"]["rom"]:
                        rom = patient["pre_op"]["rom"][side]
                        add_rom_comments(rom)

        # Post-op comments
        if "post_op" in patient:
            patient["post_op"].yaml_add_eol_comment("years", "age")
            if "rom" in patient["post_op"]:
                for side in ["right", "left"]:
                    if side in patient["post_op"]["rom"]:
                        rom = patient["post_op"]["rom"][side]
                        add_rom_comments(rom)

        # Surgery comments
        for surg in ["surgery1", "surgery2"]:
            if surg in patient and patient[surg].get("age"):
                patient[surg].yaml_add_eol_comment("years", "age")


def add_rom_comments(rom: CommentedMap):
    """Add comments to ROM fields."""
    if "hip" in rom:
        hip = rom["hip"]
        if "further_flexion" in hip:
            hip.yaml_add_eol_comment("degrees", "further_flexion")
        if "extension_thomas" in hip:
            hip.yaml_add_eol_comment("Thomas test (degrees)", "extension_thomas")
        if "extension_staheli" in hip:
            hip.yaml_add_eol_comment("Staheli test (degrees)", "extension_staheli")
        if "anteversion" in hip:
            hip.yaml_add_eol_comment("prone position (degrees)", "anteversion")

    if "knee" in rom:
        knee = rom["knee"]
        if "duncan_ely" in knee:
            knee.yaml_add_eol_comment("rectus femoris spasticity", "duncan_ely")
        if "popliteal_unilateral" in knee:
            knee.yaml_add_eol_comment("degrees", "popliteal_unilateral")

    if "ankle" in rom:
        ankle = rom["ankle"]
        if "dorsiflexion_knee90_r1" in ankle:
            ankle.yaml_add_eol_comment("R1=fast, R2=slow stretch", "dorsiflexion_knee90_r1")

    if "muscle_tone" in rom:
        mt = rom["muscle_tone"]
        if "quadriceps" in mt:
            mt.yaml_add_eol_comment("Modified Ashworth Scale", "quadriceps")


def export_to_yaml(data: dict, output_path: str, source_path: str):
    """Export data to YAML with comments."""
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.width = 120
    yaml.indent(mapping=2, sequence=4, offset=2)

    # Create root structure
    root = CommentedMap()
    root.yaml_set_start_comment(
        f"Physical Examination Data\n"
        f"Source: {source_path}\n"
        f"Generated: {date.today().isoformat()}\n"
        f"\n"
        f"Structure:\n"
        f"  - pre_op/post_op: ROM measurements before/after surgery\n"
        f"  - surgery1/surgery2: Surgical procedures and imaging studies\n"
        f"  - ROM values in degrees unless otherwise noted\n"
        f"  - muscle_tone: Modified Ashworth Scale (1-5)\n"
    )

    # Convert patients to CommentedMap
    root["patients"] = CommentedMap()
    for patient_id, patient_data in data.items():
        root["patients"][patient_id] = to_commented_map(patient_data)

    # Add field-level comments
    add_yaml_comments(root, data)

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(root, f)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Export PhysicalExamination.csv to YAML")
    parser.add_argument("--input", "-i", default="data/PhysicalExamination.csv",
                        help="Input CSV file path")
    parser.add_argument("--output", "-o", default="data/physical_examination.yaml",
                        help="Output YAML file path")
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent
    input_path = project_root / args.input
    output_path = project_root / args.output

    print(f"Parsing: {input_path}")
    patients = parse_physical_examination(str(input_path))
    print(f"Found {len(patients)} patients")

    print(f"Exporting to: {output_path}")
    export_to_yaml(patients, str(output_path), args.input)
    print("Done!")


if __name__ == "__main__":
    main()
