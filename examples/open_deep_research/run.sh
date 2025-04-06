export SERPER_API_KEY=""
export AZURE_API_KEY=""
export AZURE_API_BASE=""
export AZURE_API_VERSION=""
export HF_TOKEN="your hf token"

task_id="copy the task_id here"
task_desc=$(cat << 'EOL'
Please directly copy
the task description here
(including prompt and task)
EOL
)
output_file="outputs/$task_id.txt"
extracted_file="outputs/$task_id-extracted.txt"

/usr/bin/time -p python run.py --model-id "azure/gpt-4o" "$task_desc" 2>&1 | tee $output_file

python extract_final_answer_time.py --full_output_path $output_file --extracted_output_path $extracted_file
