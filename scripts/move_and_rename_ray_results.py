import os
import glob
import shutil

def organize_and_rename_checkpoints():
    """
    Organizes and renames checkpoint files within the Ray results directory.
    This function moves files matching 'checkpoint_0*/checkpoint-*' pattern
    to the main './ray_results/' directory and then renames them based on user input.
    """

    ray_results_dir = "./ray_results"
    
    # --- 1. Move files from ray_results/checkpoint_0*/ckpt-* to ray_results/ ---
    print(f"1. Starting file movement from 'ray_results/checkpoint_0*/ckpt-*' to '{ray_results_dir}/'...")

    # Ensure the destination directory exists
    if not os.path.exists(ray_results_dir):
        os.makedirs(ray_results_dir)
        print(f"Created directory: '{ray_results_dir}' as it did not exist.")

    moved_files_count = 0
    # Iterate through all files matching the pattern "ray_results/checkpoint_0*/ckpt-*"
    # This glob pattern directly finds paths like "ray_results/checkpoint_003000/ckpt-003000-1004_205802"
    for source_path in glob.glob("ray_results/checkpoint_0*/ckpt-*"):
        # Ensure it's a file, not a directory (though the pattern usually implies files)
        if os.path.isfile(source_path):
            file_name = os.path.basename(source_path) # Get just the file name (e.g., ckpt-003000-1004_205802)
            destination_path = os.path.join(ray_results_dir, file_name)

            try:
                shutil.move(source_path, destination_path)
                moved_files_count += 1
                # print(f"Moved '{source_path}' to '{destination_path}'") # Uncomment for verbose output
            except shutil.Error as e:
                print(f"Error moving file '{source_path}': {e}")
            
    # After moving files, try to clean up the now empty 'checkpoint_0*' subdirectories
    # We iterate through the parent directories of the moved files
    print("Attempting to clean up empty 'checkpoint_0*' subdirectories...")
    cleaned_dirs_count = 0
    for checkpoint_parent_dir in glob.glob("ray_results/checkpoint_0*"):
        if os.path.isdir(checkpoint_parent_dir):
            try:
                # Check if the directory is truly empty after file movements
                if not os.listdir(checkpoint_parent_dir): 
                    os.rmdir(checkpoint_parent_dir)
                    cleaned_dirs_count += 1
                    # print(f"Removed empty directory: '{checkpoint_parent_dir}'") # Uncomment for verbose output
            except OSError as e:
                print(f"Error removing directory '{checkpoint_parent_dir}': {e}")
                
    print(f"Total {moved_files_count} files moved to '{ray_results_dir}'.")
    print(f"Total {cleaned_dirs_count} empty 'checkpoint_0*' subdirectories removed.")
    print("-" * 50) # Separator for better readability

    # --- 2. Rename files in ./ray_results by replacing 'ckpt-' prefix with user input ---
    print("2. Starting renaming of files by replacing 'ckpt-' prefix...")

    user_prefix = input("Please enter the new file prefix (e.g., b19_da_dn2_05): ")

    renamed_files_count = 0
    # Iterate through all files in the target directory
    for filename in os.listdir(ray_results_dir):
        if filename.startswith("ckpt-"):
            old_filepath = os.path.join(ray_results_dir, filename)

            # Replace the 'ckpt-' part with the user's prefix and a hyphen
            # Using count=1 ensures only the first occurrence is replaced
            new_filename = filename.replace("ckpt-", user_prefix + "-", 1)
            new_filepath = os.path.join(ray_results_dir, new_filename)

            try:
                os.rename(old_filepath, new_filepath)
                print(f"Renamed '{filename}' to '{new_filename}'")
                renamed_files_count += 1
            except OSError as e:
                print(f"Error renaming file '{filename}': {e}")
                
    print(f"Total {renamed_files_count} files renamed.")
    print("Operation completed.")

if __name__ == "__main__":
    organize_and_rename_checkpoints()