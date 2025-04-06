import re
import sys
import os
import argparse

def extract_info(file_path):
    """
    Extract the answer and timing information from a text file.
    
    Args:
        file_path (str): Path to the text file
        
    Returns:
        tuple: (answer, real_time)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Extract the answer (everything between "Got this answer: " and "real ")
        answer_match = re.search(r'Got this answer:(.*?)(?=\nreal \d+)', content, re.DOTALL)
        answer = answer_match.group(1).strip() if answer_match else "Answer not found"
        
        # Extract timing information
        real_time_match = re.search(r'real (\d+\.\d+)', content)
        
        real_time = real_time_match.group(1) if real_time_match else "N/A"
        
        return answer, real_time
    
    except Exception as e:
        return f"Error: {str(e)}", "N/A"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full_output_path", type=str
    )
    parser.add_argument(
        "--extracted_output_path", type=str
    )
    return parser.parse_args()


def convert_to_hhmmss(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"

def main():
    args = parse_args()

    input_file = args.full_output_path
    output_file = args.extracted_output_path
    
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found")
        sys.exit(1)
    
    answer, real_time = extract_info(input_file)
    
    # Write to output file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write("EXTRACTED ANSWER:\n")
        file.write("-" * 80 + "\n")
        file.write(f"{answer}\n\n")
        file.write("TIMING INFORMATION:\n")
        file.write("-" * 80 + "\n")
        file.write(f"{convert_to_hhmmss(float(real_time))}\n")
    
    print(f"Extraction complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()