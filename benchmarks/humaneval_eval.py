"""
HumanEval: Evaluating Large Language Models Trained on Code
Mark Chen and Jerry Tworek and Heewoo Jun and Qiming Yuan and Henrique Ponde de Oliveira Pinto and Jared Kaplan and Harri Edwards and Yuri Burda and Nicholas Joseph and Greg Brockman and Alex Ray and Raul Puri and Gretchen Krueger and Michael Petrov and Heidy Khlaaf and Girish Sastry and Pamela Mishkin and Brooke Chan and Scott Gray and Nick Ryder and Mikhail Pavlov and Alethea Power and Lukasz Kaiser and Mohammad Bavarian and Clemens Winter and Philippe Tillet and Felipe Petroski Such and Dave Cummings and Matthias Plappert and Fotios Chantzis and Elizabeth Barnes and Ariel Herbert-Voss and William Hebgen Guss and Alex Nichol and Alex Paino and Nikolas Tezak and Jie Tang and Igor Babuschkin and Suchir Balaji and Shantanu Jain and William Saunders and Christopher Hesse and Andrew N. Carr and Jan Leike and Josh Achiam and Vedant Misra and Evan Morikawa and Alec Radford and Matthew Knight and Miles Brundage and Mira Murati and Katie Mayer and Peter Welinder and Bob McGrew and Dario Amodei and Sam McCandlish and Ilya Sutskever and Wojciech Zaremba 
https://arxiv.org/abs/2107.03374 https://github.com/openai/human-eval/ 
"""

import json
import logging
import multiprocessing
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Any, Tuple

from human_eval.data import HUMAN_EVAL, read_problems
from human_eval.evaluation import estimate_pass_at_k
from human_eval.execution import check_correctness  # , unsafe_execute

from . import common
from .common import HTML_JINJA
from ..types import Eval, EvalResult, SamplerBase, SamplerBaseWithId, SingleEvalResult, SingleResult, SingleProblem


def evaluate_functional_correctness(
    sample: dict[str, str],
    completions: list[str],
    n_workers: int = 4,
    timeout: float = 3.0,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    import copy

    # Check the generated samples against test suites.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for i, completion in enumerate(completions):
            args = (sample, completion, timeout, i)
            future = executor.submit(check_correctness, *args)
            futures.append(future)
        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
    passed = [int(r["passed"]) for r in results]
    return passed


class HumanEval(Eval):
    def __init__(
        self,
        num_examples: int = 250,  # restrict to a subset of the data for debugging
        num_samples_per_task: int = 5,
        ks_passes: list[int] = [1, 2, 5],
        timeout: int = 120,
    ):
        self.seed = 0
        self.examples = read_problems()
        self.examples = list(self.examples.values())

        self._num_examples = num_examples
        if self._num_examples:
            self.examples = random.Random(self.seed).sample(self.examples, num_examples)
        self._num_samples_per_task = num_samples_per_task
        self._ks_passes = ks_passes
        self._timeout = timeout

    @property
    def name(self):
        return "humaneval"

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        instruction = "Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.\n"

        def find_code(completion):
            pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
            matches = pattern.findall(completion)
            extracted_answer = matches[0] if len(matches) >= 1 else completion
            extracted_answer = extracted_answer[
                extracted_answer.find(":\n    ") + 2 :
            ]  # remove signature
            return extracted_answer

        def fn(sample: dict[str, str]):
            prompt_messages = [
                sampler._pack_message(role="user", content=instruction + sample["prompt"])
            ]

            # If sampler is a SamplerBaseWithId, we need to pass the id to the __call__ method
            if isinstance(sampler, SamplerBaseWithId):
                completions = [
                    find_code(sampler(prompt_messages, sample["task_id"])) for _ in range(self._num_samples_per_task)
                ]
            else:
                completions = [
                    find_code(sampler(prompt_messages)) for _ in range(self._num_samples_per_task)
                ]

            results = evaluate_functional_correctness(sample, completions)
            total = len(results)
            correct = sum(results)
            score = sum(results) / len(results)
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=completions[0], role="assistant"),
                score=score,
                correct_answer=[1] * len(results),
                extracted_answer=results,
            )
            convo = prompt_messages + [
                dict(content=completion, role="assistant") for completion in completions
            ]

            single_eval_result = SingleEvalResult(
                html=html,
                score=score,
                convo=convo,
                metrics={
                    f"pass@{k}": estimate_pass_at_k([total], [correct], k)
                    # this will be aggrated so no need of .mean()
                    for k in self._ks_passes
                    if total >= k
                },
            )

            # If sampler is a SamplerBaseWithId, we need to return a SingleResult
            if isinstance(sampler, SamplerBaseWithId):
                single_result = SingleResult(
                    task=self.name,
                    id=sample["task_id"],
                    problem=SingleProblem(instruction=instruction, input=sample["prompt"], target=sample["test"]),
                    output=completions,
                    answer=results,
                    score=score
                )
                return single_eval_result, single_result

            return single_eval_result

        # If sampler is a SamplerBaseWithId, we need to return a list of SingleEvalResult and SingleResult
        if isinstance(sampler, SamplerBaseWithId):
            results_tmp = common.map_with_progress(fn, self.examples, num_threads=3)
            results, single_results = zip(*results_tmp)
            return common.aggregate_results(results), single_results

        results = common.map_with_progress(fn, self.examples, num_threads=3)
        return common.aggregate_results(results)
