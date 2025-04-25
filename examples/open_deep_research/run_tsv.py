import csv
import subprocess
import os
import shlex
import sys

# --- Configuration ---
tsv_file_path = 'tasks.tsv'  # Path to your input TSV file
output_directory = 'outputs' # Directory to store output files
run_script_path = 'run.py' # Path to your main run script
extract_script_path = 'extract_final_answer_time.py' # Path to your extraction script
# model_id = 'azure/o1' # Model ID
model_id = 'azure/o3' # Model ID
time_command_path = '/usr/bin/time' # Path to the 'time' command
python_executable = sys.executable # Use the same python interpreter running this script
# ---------------------

def run_task(task_id, task_desc):
    """Runs the commands for a single task."""
    print(f"--- Starting Task: {task_id} ---")

    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    output_file = os.path.join(output_directory, f"{task_id}.txt")
    extracted_file = os.path.join(output_directory, f"{task_id}-extracted.txt")

    # --- Command 1: Run the main process with timing ---
    cmd1_list = [
        time_command_path, '-p',
        python_executable, run_script_path,
        '--model-id', model_id,
        task_desc  # Pass task_desc directly as an argument
    ]

    print(f"Executing: {' '.join(shlex.quote(arg) for arg in cmd1_list)}")
    try:
        # Open the output file for writing stdout and stderr
        with open(output_file, 'w') as f_out:
            process1 = subprocess.run(
                cmd1_list,
                stdout=f_out,
                stderr=subprocess.STDOUT, # Redirect stderr to stdout, then to file
                text=True,
                check=True # Raise an exception if the command fails
            )
        print(f"Output saved to: {output_file}")

    except FileNotFoundError as e:
         print(f"Error: Command not found - {e}. Please check paths (time, python, scripts).")
         return # Stop processing this task if command fails
    except subprocess.CalledProcessError as e:
        print(f"Error executing run.py for task {task_id}. Return code: {e.returncode}")
        print(f"Check {output_file} for details.")
        # Continue to the extraction step even if run.py failed,
        # as the output file might still exist or extraction might handle errors.
    except Exception as e:
        print(f"An unexpected error occurred during run.py execution for task {task_id}: {e}")
        return # Stop processing this task

    # --- Command 2: Extract final answer time ---
    cmd2_list = [
        python_executable, extract_script_path,
        '--full_output_path', output_file,
        '--extracted_output_path', extracted_file
    ]

    print(f"Executing: {' '.join(shlex.quote(arg) for arg in cmd2_list)}")
    try:
        process2 = subprocess.run(
            cmd2_list,
            capture_output=True, # Capture output/error streams
            text=True,
            check=True # Raise an exception if the command fails
        )
        print("Extraction successful.")
        if process2.stdout:
            print("Extraction script stdout:")
            print(process2.stdout)
        if process2.stderr:
            print("Extraction script stderr:")
            print(process2.stderr)
        print(f"Extracted data saved to: {extracted_file}")

    except FileNotFoundError as e:
         print(f"Error: Extract script not found - {e}. Please check the path.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing extract script for task {task_id}. Return code: {e.returncode}")
        print("Stderr:")
        print(e.stderr)
        print("Stdout:")
        print(e.stdout)
    except Exception as e:
        print(f"An unexpected error occurred during extraction for task {task_id}: {e}")

    print(f"--- Finished Task: {task_id} ---")
    print("-" * 30)


def main():
    """Reads the TSV and runs tasks."""
    try:
        with open(tsv_file_path, 'r', newline='', encoding='utf-8') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            # Skip header row if it exists (optional)
            # next(reader, None)
            for i, row in enumerate(reader):
                if len(row) == 2:
                    task_id, task_desc = row
                    run_task(task_id.strip(), task_desc.strip())
                else:
                    print(f"Skipping invalid row {i+1}: Expected 2 columns, found {len(row)}. Content: {row}")
    except FileNotFoundError:
        print(f"Error: Input TSV file not found at '{tsv_file_path}'")
    except Exception as e:
        print(f"An error occurred while reading the TSV file: {e}")

if __name__ == "__main__":
    main()