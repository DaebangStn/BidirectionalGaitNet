#!/usr/bin/env python3
"""
Fix skeleton symmetry by removing Y/Z axis inversion in L-side limbs.

The original skeleton uses "true local frame mirroring" where L-side limbs
have their local Y-axis pointing in the opposite direction from R-side.

This script creates a new skeleton where:
- L-side positions have only X negated (proper X-mirror)
- L-side rotation matrices use proper X-mirror formula: M * R * M where M = diag(-1,1,1)
"""

import yaml
import numpy as np
import copy
import sys

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_yaml(data, path):
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=None, allow_unicode=True, width=200)

def format_float(v, precision=4):
    """Format float with consistent precision"""
    return round(float(v), precision)

def mirror_rotation_x(R):
    """
    Compute X-mirrored rotation matrix.
    For a rotation R, the X-mirrored rotation is M * R * M where M = diag(-1, 1, 1)
    """
    R = np.array(R)
    M = np.diag([-1.0, 1.0, 1.0])
    return (M @ R @ M).tolist()

def mirror_translation_x(t):
    """Mirror translation by negating only X component"""
    return [-t[0], t[1], t[2]]

def find_r_counterpart(nodes, l_name):
    """Find the R-side counterpart for an L-side node"""
    base_name = l_name[:-1]  # Remove 'L' suffix
    r_name = base_name + 'R'
    for node in nodes:
        if node['name'] == r_name:
            return node
    return None

def fix_node_symmetry(l_node, r_node):
    """
    Fix L-side node to have proper X-mirror symmetry with R-side.
    - Translation: negate only X
    - Rotation: use M * R * M formula
    """
    # Fix body transform
    if 'body' in l_node and 'body' in r_node:
        r_body = r_node['body']
        l_body = l_node['body']

        # Mirror translation (only X)
        if 't' in r_body:
            l_body['t'] = [format_float(v) for v in mirror_translation_x(r_body['t'])]

        # Mirror rotation
        if 'R' in r_body:
            mirrored_R = mirror_rotation_x(r_body['R'])
            l_body['R'] = [[format_float(v) for v in row] for row in mirrored_R]

    # Fix joint transform
    if 'joint' in l_node and 'joint' in r_node:
        r_joint = r_node['joint']
        l_joint = l_node['joint']

        # Mirror translation (only X)
        if 't' in r_joint:
            l_joint['t'] = [format_float(v) for v in mirror_translation_x(r_joint['t'])]

        # Mirror rotation
        if 'R' in r_joint:
            mirrored_R = mirror_rotation_x(r_joint['R'])
            l_joint['R'] = [[format_float(v) for v in row] for row in mirrored_R]

        # Mirror axis for revolute joints
        if 'axis' in r_joint:
            r_axis = r_joint['axis']
            # Axis should be mirrored: negate X component
            l_joint['axis'] = [format_float(-r_axis[0]), format_float(r_axis[1]), format_float(r_axis[2])]

def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else 'data/skeleton/base.yaml'
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'data/skeleton/base_symmetric.yaml'

    print(f"Loading skeleton from: {input_path}")
    data = load_yaml(input_path)

    # Update metadata
    data['metadata']['generator'] = 'fix_skeleton_symmetry.py'
    data['metadata']['note'] = 'Fixed L/R symmetry - removed Y/Z axis inversion'

    nodes = data['skeleton']['nodes']

    # Find all L-side nodes and fix them
    l_nodes = [n for n in nodes if n['name'].endswith('L')]

    print(f"\nFound {len(l_nodes)} L-side nodes to fix:")
    for l_node in l_nodes:
        r_node = find_r_counterpart(nodes, l_node['name'])
        if r_node:
            print(f"  Fixing {l_node['name']} based on {r_node['name']}")

            # Print before values
            if 'joint' in l_node and 't' in l_node['joint']:
                print(f"    Joint t before: {l_node['joint']['t']}")
            if 'body' in l_node and 't' in l_node['body']:
                print(f"    Body t before:  {l_node['body']['t']}")

            fix_node_symmetry(l_node, r_node)

            # Print after values
            if 'joint' in l_node and 't' in l_node['joint']:
                print(f"    Joint t after:  {l_node['joint']['t']}")
            if 'body' in l_node and 't' in l_node['body']:
                print(f"    Body t after:   {l_node['body']['t']}")
        else:
            print(f"  WARNING: No R counterpart found for {l_node['name']}")

    # Save with custom formatting to match original style
    print(f"\nSaving corrected skeleton to: {output_path}")

    # Manual YAML writing to preserve formatting
    with open(output_path, 'w') as f:
        f.write("metadata:\n")
        f.write(f"  generator: \"fix_skeleton_symmetry.py\"\n")
        f.write(f"  timestamp: \"{data['metadata'].get('timestamp', 'unknown')}\"\n")
        f.write(f"  version: {data['metadata'].get('version', 'v1')}\n")
        f.write(f"  skeleton_from: \"{data['metadata'].get('skeleton_from', 'base.yaml')}\"\n")
        f.write(f"  note: \"Fixed L/R symmetry - removed Y/Z axis inversion\"\n")
        f.write(f"  git_commit: \"{data['metadata'].get('git_commit', 'unknown')}\"\n")
        f.write(f"  git_message: \"{data['metadata'].get('git_message', 'unknown')}\"\n")
        f.write("\n")
        f.write("skeleton:\n")
        f.write(f"  name: \"{data['skeleton']['name']}\"\n")
        f.write("  nodes:\n")

        for node in nodes:
            write_node(f, node)

    print("Done!")

def format_list(lst, precision=4):
    """Format a list of numbers"""
    return "[" + ", ".join(f"{v:7.4f}" for v in lst) + "]"

def format_matrix(mat):
    """Format a 3x3 matrix"""
    rows = []
    for row in mat:
        rows.append("[" + ", ".join(f"{v:7.4f}" for v in row) + "]")
    return "[" + ", ".join(rows) + "]"

def write_node(f, node):
    """Write a single node in the original YAML format"""
    name = node['name']
    parent = node.get('parent', 'None')
    ee = node.get('ee', False)

    # Start node
    f.write(f"    - {{name: {name}, parent: {parent}, ee: {str(ee).lower()}, \n")

    # Body section
    body = node['body']
    f.write(f"       body: {{type: {body['type']}, mass: {body['mass']}, ")
    f.write(f"size: {format_list(body['size'])}, ")
    f.write(f"contact: {str(body.get('contact', False)).lower()}")
    if 'obj' in body:
        f.write(f", obj: \"{body['obj']}\"")
    f.write(",\n")
    f.write(f"       R: {format_matrix(body['R'])},\n")
    f.write(f"       t: {format_list(body['t'])}}}, \n\n")

    # Joint section
    joint = node['joint']
    f.write(f"       joint: {{type: {joint['type']}")
    if 'bvh' in joint:
        f.write(f", bvh: {joint['bvh']}")
    if 'axis' in joint:
        f.write(f", axis: {format_list(joint['axis'])}")
    f.write(", \n")

    if 'lower' in joint:
        f.write(f"       lower: {format_list(joint['lower'])}, upper: {format_list(joint['upper'])},\n")
    if 'kp' in joint:
        f.write(f"       kp: {format_list(joint['kp'])}, kv: {format_list(joint['kv'])},\n")

    f.write(f"       R: {format_matrix(joint['R'])},\n")
    f.write(f"       t: {format_list(joint['t'])}}}}}\n\n")

if __name__ == '__main__':
    main()
