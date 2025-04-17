"""
Implementations mainly taken from Meta-prompting repository:
https://github.com/suzgunmirac/meta-prompting/blob/main/evaluate_outputs.py#L181

Meta-Prompting: Enhancing Language Models with Task-Agnostic Scaffolding
Mirac Suzgun and Adam Tauman Kalai
https://arxiv.org/abs/2401.12954
"""

import random
from typing import Tuple, List, Union

import requests

from . import common
from .common import HTML_JINJA
from ..types import Eval, EvalResult, MessageList, SamplerBase, SamplerBaseWithId, SingleEvalResult, SingleResult, SingleProblem
from .utils.p3_eval_helpers import execute_code_with_timeout

# Taken from the Meta-prompting repository
P3_PROMPT = """
Given a Python function "sat", the goal is to find an input or a set of inputs that makes "sat" return "True" and then include your input inside a function called "solution()".\n\nFor example, if the function was defined like this:\n\n```python\ndef sat(s: str, t:int):\n    return s == "0123456789" and t==10\n```\n\nOne correct final answer is:\n\n```python\ndef solution():\n    return "0123456789", 10\n```\n\nNow, to find a suitable input for a given "sat" function, you need to analyze the function and determine the conditions that make it return "True". Then, put your answer inside the "solution" function with that input as the argument. The final answer should be a self-contained, executable Python code containing only the answer, similar to the example above.

The sat function is defined as follows:
{input}
""".strip()

class P3Eval(Eval):
    def __init__(
        self,
        n_repeats: int = 4,
        num_examples: int | None = None,  # restrict to a subset of the data for debugging
    ):
        # Download task json from url
        task_url = "https://github.com/microsoft/PythonProgrammingPuzzles/raw/refs/heads/main/puzzles/puzzles.json"
        raw_data = requests.get(task_url).json()

        # Download the split json from url
        split_url = "https://github.com/microsoft/PythonProgrammingPuzzles/raw/refs/heads/main/puzzles/split.json"
        split_data = requests.get(split_url).json()
        test_split_keys = split_data["test"]

        # parse raw data into a list of dictionaries
        examples = []
        for id, d in enumerate(raw_data):
            # get the type of the puzzle and its id
            d_key = d["name"].split(":")[0].split("_")[0]
            d["id"] = id
            if d_key in test_split_keys:
                examples.append(d)

        print(f"P3: {len(examples)} examples")

        rng = random.Random(0)
        if num_examples:
            assert n_repeats == 1, "n_repeats only supported for num_examples = None"
            examples = rng.sample(examples, num_examples)
        examples = examples * n_repeats

        self.examples = examples
        self.n_repeats = n_repeats

    @property
    def name(self):
        return "p3"
    
    def __call__(self, sampler: Union[SamplerBase, SamplerBaseWithId]) -> Union[EvalResult, Tuple[EvalResult, List[SingleEvalResult]]]:
        def fn(row: dict):
            prompt_messages = [
                sampler._pack_message(content=P3_PROMPT.format(input=row["sat"]), role="user")
            ]
            # If sampler is a SamplerBaseWithId, we need to pass the id to the __call__ method
            if isinstance(sampler, SamplerBaseWithId):
                response_text = sampler(prompt_messages, row["id"])
            else:
                response_text = sampler(prompt_messages)

            score, extracted_answer = self.eval_for_pyton_programming_puzzles(row["sat"], response_text)
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=f"```python\n{row['sol_header']}\n{row['sol_docstring']}\n\n```",
                extracted_answer=extracted_answer,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]

            single_eval_result = SingleEvalResult(html=html, score=score, convo=convo)

            # If sampler is a SamplerBaseWithId, we need to return a SingleResult
            if isinstance(sampler, SamplerBaseWithId):
                single_result = SingleResult(
                    task=self.name,
                    id=row["id"],
                    problem=SingleProblem(instruction=P3_PROMPT.format(input=row["sat"]), input=row["sat"], target=row["sol_docstring"]),
                    output=response_text,
                    answer=extracted_answer,
                    score=score
                )
                return single_eval_result, single_result

            return single_eval_result

        # If sampler is a SamplerBaseWithId, we need to return a list of SingleEvalResult and SingleResult
        if isinstance(sampler, SamplerBaseWithId):
            results_tmp = common.map_with_progress(fn, self.examples)
            results, single_results = zip(*results_tmp)
            return common.aggregate_results(results), single_results

        results = common.map_with_progress(fn, self.examples)
        return common.aggregate_results(results)
    
    def eval_for_pyton_programming_puzzles(self, input: str, output: str) -> bool:
        """
        Take from Meta-prompting repository:
        https://github.com/suzgunmirac/meta-prompting/blob/main/evaluate_outputs.py#L181
        
        Meta-Prompting: Enhancing Language Models with Task-Agnostic Scaffolding
        Mirac Suzgun and Adam Tauman Kalai
        https://arxiv.org/abs/2401.12954
        """
        if "```python" in output:
            output = output.split("```python")[-1].strip()
            output = output.split("```")[0].strip()

        if "def sat" in output:
            if "from typing" not in output:
                output = f"from typing import *\n{output}"
            code = f"{output}\nanswer = solution()\nprint(sat(answer))"
        else:
            code = f"from typing import *\n{input}\n{output}\nanswer = solution()\nprint(sat(answer))"

        code = code.replace("List[", "list[")
        eval_bool = execute_code_with_timeout(code, timeout=3)
        if "NameError: name 'answer' is not defined" in eval_bool:
            print(f"Eval bool: {eval_bool}")
            print(f"Code:\n{code}")
            print("*" * 100)
        if "True" in eval_bool:
            return True, output
        
        return False, output