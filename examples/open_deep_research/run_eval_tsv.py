# EXAMPLE COMMAND: python examples/open_deep_research/run_gaia.py --concurrency 32 --run-name generate-traces-03-apr-noplanning --model-id gpt-4o
import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict, List, Dict

import csv
import pandas as pd
from dotenv import load_dotenv
from scripts.reformulator import prepare_response
from scripts.text_inspector_tool import TextInspectorTool
from scripts.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SimpleTextBrowser,
    VisitTool,
)
from scripts.visual_qa import visualizer
from tqdm import tqdm

from smolagents import (
    CodeAgent,
    GoogleSearchTool,
    LiteLLMModel,
    Model,
    ToolCallingAgent,
)


load_dotenv(override=True)

append_answer_lock = threading.Lock()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--model-id", type=str, default="o1")
    parser.add_argument("--run-name", type=str, required=True)
    return parser.parse_args()


class TaskRow(TypedDict):
    task_id: str
    question: str

class BrowserConfig(TypedDict):
    viewport_size: int
    downloads_folder: str
    request_kwargs: Dict[str, Any]
    serpapi_key: str | None

# Constants
TSV_FILE_PATH: str = 'tasks.tsv'
OUTPUT_DIRECTORY: str = 'outputs'
USER_AGENT: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"

BROWSER_CONFIG: BrowserConfig = {
    "viewport_size": 1024 * 5,
    "downloads_folder": "downloads_folder",
    "request_kwargs": {
        "headers": {"User-Agent": USER_AGENT},
        "timeout": 300,
    },
    "serpapi_key": os.getenv("SERPAPI_API_KEY"),
}

os.makedirs(f"./{BROWSER_CONFIG['downloads_folder']}", exist_ok=True)
# ---------------------


def create_agent_team(model: Model):
    text_limit = 100000
    ti_tool = TextInspectorTool(model, text_limit)

    browser = SimpleTextBrowser(**BROWSER_CONFIG)

    WEB_TOOLS = [
        GoogleSearchTool(provider="serper"),
        VisitTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
        TextInspectorTool(model, text_limit),
    ]

    text_webbrowser_agent = ToolCallingAgent(
        model=model,
        tools=WEB_TOOLS,
        max_steps=20,
        verbosity_level=2,
        planning_interval=4,
        name="search_agent",
        description="""A team member that will search the internet to answer your question.
    Ask him for all your questions that require browsing the web.
    Provide him as much context as possible, in particular if you need to search on a specific timeframe!
    And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
    Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
    """,
        provide_run_summary=True,
    )
    text_webbrowser_agent.prompt_templates["managed_agent"]["task"] += """You can navigate to .txt online files.
    If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
    Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information."""

    manager_agent = CodeAgent(
        model=model,
        tools=[visualizer, ti_tool],
        max_steps=12,
        verbosity_level=2,
        additional_authorized_imports=["*"],
        planning_interval=4,
        managed_agents=[text_webbrowser_agent],
    )
    return manager_agent


def append_answer(entry: Dict[str, Any], jsonl_file_str: str) -> None:
    jsonl_file = Path(jsonl_file_str)
    jsonl_file.parent.mkdir(parents=True, exist_ok=True)
    with append_answer_lock, open(jsonl_file, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry) + "\n")
    assert os.path.exists(jsonl_file), "File not found!"
    print(f"Answer exported to file: {jsonl_file.resolve()}")


def answer_single_question(example: TaskRow, model_id: str, answers_file: str):
    model_params: dict[str, Any] = {
        "model_id": model_id,
        "custom_role_conversions": {"tool-call": "assistant", "tool-response": "user"},
    }
    if model_id in ["o1", "o3"]:
        model_params["reasoning_effort"] = "high"
        model_params["max_completion_tokens"] = 8192
    else:
        model_params["max_tokens"] = 4096
    model = LiteLLMModel(**model_params)
    # model = InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct", provider="together", max_tokens=4096)

    agent = create_agent_team(model)

    augmented_question = """You have one question to answer. It is paramount that you provide a correct answer.
Give it all you can: I know for a fact that you have access to all the relevant tools to solve it and find the correct answer (the answer does exist). Failure or 'I cannot answer' or 'None found' will not be tolerated, success will be rewarded.
Run verification steps if that's needed, you must make sure you find the correct answer!
Here is the task:
"""
    augmented_question += example["question"]


    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        # Run agent 🚀
        final_result = agent.run(augmented_question)

        agent_memory = agent.write_memory_to_messages()

        final_result = prepare_response(augmented_question, agent_memory, reformulation_model=model)

        output = str(final_result)
        for memory_step in agent.memory.steps:
            memory_step.model_input_messages = None
        intermediate_steps = agent_memory

        # Check for parsing errors which indicate the LLM failed to follow the required format
        parsing_error = True if any(["AgentParsingError" in step for step in intermediate_steps]) else False

        # check if iteration limit exceeded
        iteration_limit_exceeded = True if "Agent stopped due to iteration limit or time limit." in output else False
        raised_exception = False

    except Exception as e:
        print("Error on ", augmented_question, e)
        output = None
        intermediate_steps = []
        parsing_error = False
        iteration_limit_exceeded = False
        exception = e
        raised_exception = True

    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    token_counts_manager = agent.monitor.get_total_token_counts()
    token_counts_web = list(agent.managed_agents.values())[0].monitor.get_total_token_counts()
    total_token_counts = {
        "input": token_counts_manager["input"] + token_counts_web["input"],
        "output": token_counts_manager["output"] + token_counts_web["output"],
    }
    annotated_example = {
        "agent_name": model.model_id,
        "question": example["question"],
        "augmented_question": augmented_question,
        "prediction": output,
        "intermediate_steps": intermediate_steps,
        "parsing_error": parsing_error,
        "iteration_limit_exceeded": iteration_limit_exceeded,
        "agent_error": str(exception) if raised_exception else None,
        "task_id": example["task_id"],
        # "true_answer": example["true_answer"], # TODO: Add true answer to tsv file
        "start_time": start_time,
        "end_time": end_time,
        "token_counts": total_token_counts,
    }
    append_answer(annotated_example, answers_file)


def get_examples_to_answer(answers_file: str, rows: List[TaskRow]) -> List[TaskRow]:
    print(f"Loading answers from {answers_file}...")
    try:
        os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
        questions_answered = pd.read_json(answers_file, lines=True)
    except Exception as e:
        print("Error when loading records: ", e)
        print("No usable records! ▶️ Starting new.")
        questions_answered = None

    if questions_answered is None or questions_answered.empty:
        return rows

    # Filter out rows that have already been answered.
    # Please manually remove (or use a custom script to remove) all questions
    # that have been answered but not answered correctly
    # (i.e., where questions_answered['prediction'] is None 
    # or questions_answered['agent_error'] is not None),
    # to allow the script to run again and answer the remaining questions.
    rows_to_answer = []
    for row in rows:
        if row["question"] not in questions_answered['question'].values: 
            rows_to_answer.append(row)
    return rows_to_answer

def read_tsv() -> List[TaskRow]:
    rows: List[TaskRow] = []
    with open(TSV_FILE_PATH, 'r', newline='', encoding='utf-8') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            if len(row) != 2:
                raise ValueError(f"Invalid row: {row}")
            else:
                rows.append({"task_id": row[0], "question": row[1]}) # TODO: Add true answer to tsv file
    return rows

def main():
    args = parse_args()
    print(f"Starting run with arguments: {args}")

    rows = read_tsv()
    answers_file = f"{OUTPUT_DIRECTORY}/{args.run_name}.jsonl"
    tasks_to_run = get_examples_to_answer(answers_file, rows)

    with ThreadPoolExecutor(max_workers=args.concurrency) as exe:
        futures = [
            exe.submit(answer_single_question, example, args.model_id, answers_file)
            for example in tasks_to_run
        ]
        for f in tqdm(as_completed(futures), total=len(tasks_to_run), desc="Processing tasks"):
            try:
                f.result()
            except Exception as e:
                print(f"Error ThreadPoolExecutor : {e}")

    print("All tasks processed.")


if __name__ == "__main__":
    main()
