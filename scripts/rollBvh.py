"""
BVH Frame Roller
This script rolls/shifts the frames in a BVH motion capture file.
Useful for changing the starting frame of a cyclic motion.
"""

from pathlib import Path
from bvh import Bvh


rolling_frame_index = 39
bvh_file_path = "data/motion/walk.bvh"


def format_frame_values(frame):
    """
    Format all values in a frame to 5 decimal places.

    Args:
        frame: List of string values representing frame data

    Returns:
        List of formatted string values with 5 decimal places
    """
    return [f"{float(value):.5f}" for value in frame]


def roll_bvh_frames(input_path, output_path, frame_offset):
    """
    Roll frames in a BVH file by the specified offset.
    
    Args:
        input_path: Path to input BVH file
        output_path: Path to output BVH file
        frame_offset: Number of frames to shift (positive = shift forward)
    """
    print(f"🔄 Rolling BVH frames...")
    print(f"   Input file: {input_path}")
    print(f"   Frame offset: {frame_offset}")
    
    # Read the original BVH file content
    try:
        with open(input_path, 'r') as f:
            print(f"   ✓ Reading BVH file...")
            bvh_content = f.read()
            mocap = Bvh(bvh_content)
    except FileNotFoundError:
        print(f"   ✗ Error: File not found: {input_path}")
        return False
    except Exception as e:
        print(f"   ✗ Error reading file: {e}")
        return False
    
    # Get frame data
    frames = mocap.frames
    frame_time = mocap.frame_time
    total_frames = len(frames)
    
    print(f"   ✓ Loaded {total_frames} frames (frame time: {frame_time}s)")
    
    # Validate offset
    if abs(frame_offset) >= total_frames:
        print(f"   ⚠ Warning: Offset {frame_offset} is larger than total frames {total_frames}")
        frame_offset = frame_offset % total_frames
        print(f"   → Adjusted offset to: {frame_offset}")
    
    # Reorder frames
    frames_b = frames[frame_offset:]  # sequence B (will be at start)
    frames_a = frames[:frame_offset]  # sequence A (will be at end)

    # Calculate position offset from last frame of sequence B to first frame of sequence A
    # Root position is in the first 3 channels (Xposition, Yposition, Zposition)
    last_frame_b = frames_b[-1]
    first_frame_a = frames_a[0]

    # Calculate offset (last_B - first_A)
    offset_x = float(last_frame_b[0]) - float(first_frame_a[0])
    offset_y = float(last_frame_b[1]) - float(first_frame_a[1])
    offset_z = float(last_frame_b[2]) - float(first_frame_a[2])

    print(f"   📐 Position offset: X={offset_x:.5f}, Y={offset_y:.5f}, Z={offset_z:.5f}")

    # Apply offset to all frames in sequence A
    frames_a_adjusted = []
    for frame in frames_a:
        adjusted_frame = frame.copy()
        adjusted_frame[0] = str(float(frame[0]) + offset_x)
        adjusted_frame[1] = str(float(frame[1]) + offset_y)
        adjusted_frame[2] = str(float(frame[2]) + offset_z)
        frames_a_adjusted.append(adjusted_frame)

    # Combine: B + adjusted A
    reordered = frames_b + frames_a_adjusted
    print(f"   ✓ Reordered frames with continuity fix (new start: frame {frame_offset})")
    
    # Write output file
    try:
        # Parse the original file to extract header
        lines = bvh_content.split('\n')
        header_lines = []
        motion_line_idx = 0
        
        # Find the MOTION section
        for i, line in enumerate(lines):
            if line.strip() == "MOTION":
                motion_line_idx = i
                header_lines = lines[:i+1]  # Include MOTION line
                break
        
        if not header_lines:
            print(f"   ✗ Error: Could not find MOTION section in BVH file")
            return False
        
        with open(output_path, 'w') as f:
            # Write header (everything up to and including MOTION)
            for line in header_lines:
                f.write(line + '\n')
            
            # Write frame info
            f.write(f"Frames: {len(reordered)}\n")
            f.write(f"Frame Time: {frame_time}\n")
            
            # Write frame data
            for frame in reordered:
                formatted_frame = format_frame_values(frame)
                f.write(" ".join(formatted_frame) + "\n")
        
        print(f"   ✓ Successfully wrote output file: {output_path}")
        print(f"✅ Done!\n")
        return True
        
    except Exception as e:
        print(f"   ✗ Error writing output file: {e}")
        return False


if __name__ == "__main__":    
    # Generate output filename
    input_path = Path(bvh_file_path)
    output_filename = f"{input_path.stem}_rolled{rolling_frame_index}{input_path.suffix}"
    output_path = input_path.parent / output_filename
    
    # Roll the frames
    success = roll_bvh_frames(
        input_path=bvh_file_path,
        output_path=str(output_path),
        frame_offset=rolling_frame_index
    )
    
    if not success:
        exit(1)
