"""
Implementations mainly taken from Meta-prompting repository:
https://github.com/suzgunmirac/meta-prompting/blob/main/evaluate_outputs.py#L181

Meta-Prompting: Enhancing Language Models with Task-Agnostic Scaffolding
Mirac Suzgun and Adam Tauman Kalai
https://arxiv.org/abs/2401.12954
"""

import json
import random
from typing import Tuple, List, Union

import pandas as pd

from . import common
from .common import HTML_JINJA
from ..types import Eval, EvalResult, MessageList, SamplerBase, SamplerBaseWithId, SingleEvalResult, SingleResult, SingleProblem
from .utils.sonnet_eval_helpers import sonnet_errors

# Taken from the Meta-prompting repository
SONNET_PROMPT = """
Write a sonnet that adheres strictly to the specified rhyme scheme and includes the given words.

{input}

{answer_pattern}
""".strip()

class SonnetEval(Eval):
    def __init__(
        self,
        n_repeats: int = 4,
        num_examples: int | None = None,  # restrict to a subset of the data for debugging
    ):
        # Init answer pattern
        self.answer_pattern = common.DefaultParser()

        # Download task json from url
        task_url = "https://github.com/suzgunmirac/meta-prompting/raw/refs/heads/main/data/Sonnets-Standard.jsonl"
        df = pd.read_json(task_url, lines=True)

        # Add id column
        df["id"] = df.index

        # Add answer pattern column
        df['answer_pattern'] = self.answer_pattern.answer_pattern

        examples = [row.to_dict() for _, row in df.iterrows()]

        print(f"Sonnet: {len(examples)} examples")

        rng = random.Random(0)
        if num_examples:
            assert n_repeats == 1, "n_repeats only supported for num_examples = None"
            examples = rng.sample(examples, num_examples)
        examples = examples * n_repeats

        self.examples = examples
        self.n_repeats = n_repeats

    @property
    def name(self):
        return "sonnet-writing"
    
    def __call__(self, sampler: Union[SamplerBase, SamplerBaseWithId]) -> Union[EvalResult, Tuple[EvalResult, List[SingleEvalResult]]]:
        def fn(row: dict):
            prompt_messages = [
                sampler._pack_message(
                    content=SONNET_PROMPT.format(**row),
                    role="user"
                )
            ]
            # If sampler is a SamplerBaseWithId, we need to pass the id to the __call__ method
            if isinstance(sampler, SamplerBaseWithId):
                response_text = sampler(prompt_messages, row["id"])
            else:
                response_text = sampler(prompt_messages)

            extracted_answer = self.answer_pattern.parse(response_text)
            score = self.check_sonnet_errors(extracted_answer, row["target"])
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
                    task=self.name,
                    id=row["id"],
                    problem=SingleProblem(
                        instruction=SONNET_PROMPT.format(**row),
                        input=row["input"],
                        target=row["target"]
                    ),
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
    
    def check_sonnet_errors(self, output: str, rhyme_scheme: str) -> bool:
        try:
            errors = sonnet_errors(output, rhyme_scheme)
            if not errors:
                return True
            return False
        except Exception as e:
            return False