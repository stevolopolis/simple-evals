import os
import datetime
import json

from typer import Typer

from .models.router import *

from .benchmarks import common
from .benchmarks.browsecomp_eval import BrowseCompEval
from .benchmarks.drop_eval import DropEval
from .benchmarks.gpqa_eval import GPQAEval
from .benchmarks.humaneval_eval import HumanEval
from .benchmarks.math_eval import MathEval
from .benchmarks.mgsm_eval import MGSMEval
from .benchmarks.mmlu_eval import MMLUEval
from .benchmarks.simpleqa_eval import SimpleQAEval
from .benchmarks.gameof24_eval import Gameof24Eval
from .sampler.chat_completion_sampler import (
    ChatCompletionSampler,
)

from dotenv import load_dotenv

load_dotenv()


app = Typer()


grading_sampler = ChatCompletionSampler(model="gpt-4o")
equality_checker = ChatCompletionSampler(model="gpt-4-turbo-preview")

n_seeds = 2

def get_evals(eval_name, debug_mode):
    num_examples = 5 if debug_mode else None
    # Set num_examples = None to reproduce full evals
    match eval_name:
        case "mmlu":
            return MMLUEval(num_examples=1 if debug_mode else num_examples)
        case "math":
            return MathEval(
                equality_checker=equality_checker,
                num_examples=num_examples,
                n_repeats=1 if debug_mode else n_seeds,
            )
        case "math500": 
            return MathEval(
                equality_checker=equality_checker,
                num_examples=num_examples,
                n_repeats=1 if debug_mode else n_seeds,
                split="math_500_test"
            )
        case "gpqa":
            return GPQAEval(
                n_repeats=1 if debug_mode else n_seeds, 
                num_examples=num_examples
            )
        case "mgsm":
            return MGSMEval(num_examples_per_lang=10 if debug_mode else 250)
        case "drop":
            return DropEval(
                num_examples=num_examples if num_examples else 10,
                train_samples_per_prompt=3,
            )
        case "humaneval":
            return HumanEval(num_examples=num_examples if num_examples else 10)
        case "simpleqa":
            return SimpleQAEval(
                grader_model=grading_sampler,
                num_examples=num_examples if num_examples else 10,
            )
        case "browsecomp":
            return BrowseCompEval(
                grader_model=grading_sampler,
                num_examples=num_examples if num_examples else 10,
            )
        case "gameof24":
            return Gameof24Eval(
                num_examples=num_examples,
                n_repeats=1 if debug_mode else n_seeds,
            )
        case _:
            raise Exception(f"Unrecognized eval type: {eval_name}")


@app.command()
def run_benchmark(
    eval_name: str = 'math500',
    model_id: str = 'openai/gpt-4o-mini',
    debug_mode: bool = True
):
    model_tags = model_id.split('/')
    if len(model_tags) == 2:
        model_provider, model_name = model_tags
        method = "vanilla"
    elif len(model_tags) == 3:
        model_provider, model_name, method = model_tags
        model_id = f"{model_provider}/{model_name}"
    else:
        raise ValueError(f'Invalid model id: {model_id}')
    
    output_dir = f'benchmark_results/{model_name}/{method}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get sampler
    sampler = get_model_from_id(model_id, method)

    # Get evaluator
    eval_obj = get_evals(eval_name, debug_mode)

    # Run evaluator
    result, single_results = eval_obj(sampler)

    # Create results directory
    output_dir = f'benchmark_results/{model_name}/{method}'
    output_agg_dir = f'benchmark_results/aggregated/{model_name}/{method}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_agg_dir):  
        os.makedirs(output_agg_dir)

    # Get filenames
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d-%H:%M:%S")
    report_filename = f"{output_agg_dir}/{eval_name}_{timestamp_str}.html"
    result_filename = f"{output_dir}/{eval_name}_{timestamp_str}.jsonl"
    trace_filename = f"{output_dir}/{eval_name}_trace_{timestamp_str}.json"

    # Write report html (contains score, metrics, and examples)
    print(f"Writing report to {report_filename}")
    with open(report_filename, "w") as fh:
        fh.write(common.make_report(result))

    print(result.score, result.metrics)

    # Write single results to jsonl
    with open(result_filename, "w") as fh:
        for single_result in single_results:
            # convert single_result to dict
            single_result_dict = single_result.__dict__
            single_result_dict["problem"] = single_result.problem.__dict__
            fh.write(json.dumps(single_result_dict) + "\n")

    # Save trace
    sampler.save_trace(trace_filename)

@app.command()
def run_all_methods_same_model(
    task: str = 'gameof24',
    model: str = 'openai/gpt-4o',
    debug: bool = False
):
    for method in get_all_methods():
        if method == "cot":
            continue    
        model_id = model + "/" + method
        run_benchmark(task, model_id, debug)


@app.command()
def run_all_models_same_method(
    task: str = 'gameof24',
    method: str = 'cot',
    debug: bool = False
):
    models = [
        # "openai/gpt-4o",
        # "openai/gpt-4o-mini",
        "openai/gpt-4.1",
        "openai/gpt-4.1-mini",
        # "openai/gpt-4.1-nano",
        # "openai/o1-mini",
        # "openai/o1"
    ]

    for model in models:
        model_id = model + "/" + method
        run_benchmark(task, model_id, debug)




if __name__ == "__main__":
    app()