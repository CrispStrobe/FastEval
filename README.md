**Project status:** 
The original creator of this project [FastEval/FastEval](https://github.com/FastEval/FastEval) wrote: "I will not add significant new features and mostly fix bugs."
This fork includes MT-Bench-de as included by [mayflower/FastEval](https://github.com/mayflower/FastEval) and adds a few fixes for local inference flexibility, i.a.

# FastEval

This project allows you to quickly evaluate instruction-following and chat language models on a number of benchmarks.
See the [comparison to lm-evaluation-harness](docs/comparison-to-lm-eval.md) for more information.
There is also a [leaderboard](https://fasteval.github.io/FastEval/).

## Features

- **Evaluation on various benchmarks with a single command.** Supported benchmarks are [MT‑Bench](https://arxiv.org/abs/2306.05685) for conversational capabilities, [HumanEval+](https://github.com/evalplus/evalplus) and [DS-1000](https://ds1000-code-gen.github.io/) for Python coding performance, Chain of Thought (GSM8K + MATH + BBH + MMLU + AGIEVAL) for reasoning capabilities as well as [custom test data](docs/custom-test-data.md).
- **High performance.** FastEval uses [vLLM](https://github.com/vllm-project/vllm) for fast inference by default and can also optionally make use of [text-generation-inference](https://github.com/huggingface/text-generation-inference). Both methods are ~20x faster than using huggingface transformers.
- **Detailed information about model performance.** FastEval saves the outputs of the language model and other intermediate results to disk. This makes it possible to get deeper insight into model performance. You can look at the [performance on different categories](https://fasteval.github.io/FastEval/#?benchmark=mt-bench) and even inspect [individual model outputs](https://fasteval.github.io/FastEval/#?benchmark=cot&task=bbh/date_understanding&id=eb74c9e1-8836-4c3a-8f50-a25808d20eee).
- **Use of model-specific prompt templates**: FastEval uses the right prompt template depending on the evaluated model. Many prompt templates are supported and the use of [FastChat](https://github.com/lm-sys/FastChat) expands this even further.

## Installation

```bash
# Install `python3.10`, `python3.10-venv` and `python3.10-dev`.
# The following command assumes an ubuntu >= 22.04 system.
apt install python3.10 python3.10-venv python3.10-dev

# Clone this repository, make it the current working directory
git clone --depth 1 https://github.com/FastEval/FastEval.git
cd FastEval

# Set up the virtual environment
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This already installs [vLLM](https://github.com/vllm-project/vllm) for fast inference which is usually enough [for most models](https://vllm.readthedocs.io/en/latest/models/supported_models.html). However, if you encounter any problems with vLLM or your model is not supported, FastEval also supports using [text-generation-inference](https://github.com/huggingface/text-generation-inference) as an alternative. Please see [here](docs/text-generation-inference.md) if you would like to use text-generation-inference.

### OpenAI API Key for LLM-as-a-judge

[MT-Bench](https://arxiv.org/abs/2306.05685) uses an LLM as a judge for evaluating model outputs.
For this benchmark, you can configure the following environment variables:
* MT_BENCH_JUDGE_TYPE=openai_judge
* MT_BENCH_JUDGE_MODEL=gpt-4-0613
* JUDGE_API_BASE=https://api.openai.com/v1
* JUDGE_API_KEY=your_api_key
Note that methods other than setting this environment variable won't work.
The cost of evaluating a new model on MT-Bench is approximately $5.

## Evaluation

⚠️ Running `fasteval` currently executes untrusted code from models with remote code as well as LLM generated code when using [HumanEval+](https://github.com/evalplus/evalplus) and [DS-1000](https://ds1000-code-gen.github.io/). Please note that there is currently no integrated sandbox.

To evaluate a new model, call `fasteval` in the following way:
```
./fasteval [-b <benchmark_name_1>...] -t model_type -m model_name
````

The `-b` flag specifies the benchmarks that you want to evaluate your model on. The default is `all`, but you can also specify one or multiple individual benchmarks. Possible values are [`mt-bench`](https://fasteval.github.io/FastEval/#?benchmark=mt-bench), `mt-bench-de`, `mt-bench-vago`, [`human-eval-plus`](https://fasteval.github.io/FastEval/#?benchmark=human-eval-plus), [`ds1000`](https://fasteval.github.io/FastEval/#?benchmark=ds1000), [`cot`](https://fasteval.github.io/FastEval/#?benchmark=cot), `cot/gsm8k`, `cot/math`, `cot/bbh`, `cot/mmlu` and [`custom-test-data`](docs/custom-test-data.md).

The `-t` flag specifies the type of the model which is either the prompt template or the API client that will be used. [Please see here](docs/model-type.md) for information on which model type to select for your model.

The `-m` flag specifies the name of the model which can be a path to a model on huggingface, a local folder or an OpenAI model name.

For example, this command will evaluate [`OpenAssistant/pythia-12b-sft-v8-2.5k-steps`](https://huggingface.co/OpenAssistant/pythia-12b-sft-v8-2.5k-steps) on [HumanEval+](https://fasteval.github.io/FastEval/#?benchmark=human-eval-plus):
```bash
./fasteval -b human-eval-plus -t open-assistant -m OpenAssistant/pythia-12b-sft-v8-2.5k-steps
```

For local inference per OpenAI compatible API, you can set the following environment variables:
OPENAI_API_BASE=http://localhost:8000/v1
OPENAI_API_KEY=ollama
This way, you can e.g. use llama_cpp.server (default: http://localhost:8000/v1) or ollama (default: http://localhost:11434/v1)

There are also flags available for enabling & configuring data parallel evaluation, setting model arguments and changing the inference backend. Please use `./fasteval -h` for more information.

## Viewing the results

A very short summary of the final scores will be written to stdout after the evaluation has finished.

More details are available through the web UI where you can view performance on different subtasks or inspect individual model inputs & outputs.
To access the web UI, use `python3 -m http.server` in the root folder of this repository.
This will start a simple webserver for static files.
The server usually runs on port `8000` in which case you can view the detailed results at [localhost:8000](http://localhost:8000).

## Help & Contributing

For questions, problems and contributions, join the [Alignment Lab AI discord server](https://discord.gg/ad27GQgc7K) or create a github issue.
Contributions are **greatly welcome**.
Please read the [contributing guide](.github/CONTRIBUTING.md) for more information.
