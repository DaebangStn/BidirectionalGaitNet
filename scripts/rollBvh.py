"""
BVH Frame Roller
This script rolls/shifts the frames in a BVH motion capture file.
Useful for changing the starting frame of a cyclic motion.
"""

from pathlib import Path
from bvh import Bvh


rolling_frame_index = 119
bvh_file_path = "data/motion/walk.bvh"


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
    reordered = frames[frame_offset:] + frames[:frame_offset]
    print(f"   ✓ Reordered frames (new start: frame {frame_offset})")
    
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
                f.write(" ".join(frame) + "\n")
        
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
