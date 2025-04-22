"""
BBEH: BIG-Bench Extra Hard
Mehran Kazemi, Bahare Fatemi, Hritik Bansal, John Palowitch, Chrysovalantis Anastasiou
https://arxiv.org/abs/2502.19187
"""

import random
from typing import Tuple, List, Union

import requests

from . import common
from .common import HTML_JINJA
from ..types import Eval, EvalResult, MessageList, SamplerBase, SamplerBaseWithId, SingleEvalResult, SingleResult, SingleProblem
from .utils.bbeh_eval_helpers import evaluate_correctness

# Default in BBEH (reproducibility statement in arxiv paper appendix)
BBEH_SUFFIX_WITH_COT = """
Think step by step, and when you provide the final answer, please use the prefix "The answer is:"
without any modification, and provide the answer directly, with no formatting, no bolding, and
no markup. For instance: "The answer is: 42" or "The answer is: yes". If the question is multiple
choice with a single correct answer, the final answer must only be the letter corresponding to
the correct answer. For example, "The answer is: (a)".
"""

BBEH_SUFFIX_WITHOUT_COT = """
When you provide the final answer, please use the prefix "The answer is:"
without any modification, and provide the answer directly, with no formatting, no bolding, and
no markup. For instance: "The answer is: 42" or "The answer is: yes". If the question is multiple
choice with a single correct answer, the final answer must only be the letter corresponding to
the correct answer. For example, "The answer is: (a)".
"""

BBEH_SUFFIX = BBEH_SUFFIX_WITH_COT

BBEH_URL = "https://github.com/google-deepmind/bbeh/raw/refs/heads/main/bbeh/benchmark_tasks"


class BBEHEval(Eval):
    def __init__(
        self,
        subtask: str,
        n_repeats: int = 4,
        num_examples: int | None = None,  # restrict to a subset of the data for debugging
    ):
        if subtask is None:
            raise ValueError(f"Subtask must be provided. Available subtasks are:\n{self.subtasks}")
    
        assert subtask in self.subtasks, f"Provided subtask is not valid. Available subtasks are:\n{self.subtasks}"
        
        self.task_name = f"bbeh-{subtask}"
        
        # Download task json from url
        task_url = f"{BBEH_URL}/bbeh_{subtask}/task.json"
        raw_data = requests.get(task_url).json()

        # Parse "examples" field for list of data
        examples = raw_data["examples"]

        # Add id column
        examples = [example | {"id": i} for i, example in enumerate(examples)]

        print(f"BBEH: {self.task_name} ({len(examples)} examples)")

        rng = random.Random(0)
        if num_examples:
            assert n_repeats == 1, "n_repeats only supported for num_examples = None"
            examples = rng.sample(examples, num_examples)
        examples = examples * n_repeats

        self.examples = examples
        self.n_repeats = n_repeats

    @property
    def name(self):
        return "bbeh"
    
    @property
    def subtasks(self):
        return [
            'boardgame_qa',
            'boolean_expressions',
            'buggy_tables',
            'causal_understanding',
            'disambiguation_qa',
            'dyck_languages',
            'geometric_shapes',
            'hyperbaton',
            'linguini',
            'movie_recommendation',
            'multistep_arithmetic',
            'nycc',
            'object_counting',
            'object_properties',
            'sarc_triples',
            'shuffled_objects',
            'spatial_reasoning',
            'sportqa',
            'temporal_sequence',
            'time_arithmetic',
            'web_of_lies',
            'word_sorting',
            'zebra_puzzles'
        ]

    def __call__(self, sampler: Union[SamplerBase, SamplerBaseWithId]) -> Union[EvalResult, Tuple[EvalResult, List[SingleEvalResult]]]:
        def fn(row: dict):
            prompt_messages = [
                sampler._pack_message(content=row["input"] + BBEH_SUFFIX, role="user")
            ]
            # If sampler is a SamplerBaseWithId, we need to pass the id to the __call__ method
            if isinstance(sampler, SamplerBaseWithId):
                response_text = sampler(prompt_messages, row["id"])
            else:
                response_text = sampler(prompt_messages)

            score, extracted_answer = evaluate_correctness(response_text, row["target"], return_extracted_answer=True)
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["target"],
                extracted_answer=extracted_answer,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]

            single_eval_result = SingleEvalResult(html=html, score=score, convo=convo)

            # If sampler is a SamplerBaseWithId, we need to return a SingleResult
            if isinstance(sampler, SamplerBaseWithId):
                single_result = SingleResult(
                    task=self.task_name,
                    id=row["id"],
                    problem=SingleProblem(instruction=row["input"] + BBEH_SUFFIX, input=row["input"], target=row["target"]),
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
    
    def eval_fn(self, sample, reference, return_extracted_answer=False):
        return evaluate_correctness(sample, reference, return_extracted_answer=return_extracted_answer)