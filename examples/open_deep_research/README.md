# Open Deep Research

Welcome to this open replication of [OpenAI's Deep Research](https://openai.com/index/introducing-deep-research/)! This agent attempts to replicate OpenAI's model and achieve similar performance on research tasks.

Read more about this implementation's goal and methods in our [blog post](https://huggingface.co/blog/open-deep-research).


This agent achieves **55% pass@1** on the GAIA validation set, compared to **67%** for the original Deep Research.

## Setup

To get started, follow the steps below:

### Clone the repository

```bash
git clone https://github.com/nnnyt/smolagents.git
cd smolagents/examples/open_deep_research
```

### Install dependencies

Run the following command to install the required dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Install the development version of `smolagents`

```bash
pip install -e ../../.[dev]
```

### Set up environment variables

The agent uses the `GoogleSearchTool` for web search, which requires an environment variable with the corresponding API key, based on the selected provider:
- `SERPAPI_API_KEY` for SerpApi: [Sign up here to get a key](https://serpapi.com/users/sign_up)
- `SERPER_API_KEY` for Serper: [Sign up here to get a key](https://serper.dev/signup)

Depending on the model you want to use, you may need to set environment variables.
For example, to use the default `o1` model, you need to set the `OPENAI_API_KEY` environment variable.
[Sign up here to get a key](https://platform.openai.com/signup).

> [!WARNING]
> The use of the default `o1` model is restricted to tier-3 access: https://help.openai.com/en/articles/10362446-api-access-to-o1-and-o3-mini

If you use Azure for OpenAI models, you should set those environment variables for Azure.

## Usage

Then you're good to go! 

1. Save the samples in tasks.tsv. In this TSV file, each sample occupies one line without a table header. Each line should be formatted as f`{task_id}\t{task_description}`, where `task_id` can be any string.
2. Run `run_tsv.py`. You can set the model name in this program.

### Alternative usage

If you don't want to use this TSV file for many samples but only would like to test one sample, you can also run in an alternative.

Run the run.py script, as in:
```bash
python run.py --model-id "o1" "Your question here!"
```
