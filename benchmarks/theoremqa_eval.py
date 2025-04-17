"""
Implementations mainly taken from Meta-prompting repository:
https://github.com/suzgunmirac/meta-prompting/blob/main/evaluate_outputs.py#L181

Meta-Prompting: Enhancing Language Models with Task-Agnostic Scaffolding
Mirac Suzgun and Adam Tauman Kalai
https://arxiv.org/abs/2401.12954
"""

import base64
import random
import io
from typing import Tuple, List, Union

from datasets import load_dataset

from . import common
from .common import HTML_JINJA, DefaultParser
from ..types import Eval, EvalResult, MessageList, SamplerBase, SamplerBaseWithId, SingleEvalResult, SingleResult, SingleProblem

from .utils.theoremqa_eval_helpers import compare_answer_with_groundtruth, answer_clean_with_parser

# Taken from the Meta-prompting repository
THEOREMQA_DEFAULT_PROMPT = """
Your task is to answer the following question.

Problem: {problem}

{answer_format}
""".strip()

THEOREM_PROMPT = """
To solve this problem, you may use the following theorem:
{theorem}
""".strip()


class TheoremQAEval(Eval):
    def __init__(
        self,
        n_repeats: int = 4,
        num_examples: int | None = None,  # restrict to a subset of the data for debugging
    ):
        # Setup answer parser
        self.answer_parser = DefaultParser()

        # Download dataset from huggingface 
        dataset = load_dataset("TIGER-Lab/TheoremQA")
        examples = [dict(row) | {"id": i} for i, row in enumerate(dataset["test"])]

        print(f"TheoremQA: {len(examples)} examples")

        rng = random.Random(0)
        if num_examples:
            assert n_repeats == 1, "n_repeats only supported for num_examples = None"
            examples = rng.sample(examples, num_examples)
        examples = examples * n_repeats

        self.examples = examples
        self.n_repeats = n_repeats

    @property
    def name(self):
        return "theoremqa"
    
    def __call__(self, sampler: Union[SamplerBase, SamplerBaseWithId]) -> Union[EvalResult, Tuple[EvalResult, List[SingleEvalResult]]]:
        def fn(row: dict):
            # Prepare the question content
            content = [
                sampler._handle_text(THEOREMQA_DEFAULT_PROMPT.format(problem=row["Question"], answer_format=self.answer_parser.answer_pattern))
            ]
            # If the question includes an image, we need to add it to the prompt
            if row["Picture"] is not None:
                # Convert the PIL image to a base64 string
                buffered = io.BytesIO()
                row["Picture"].save(buffered, format="PNG")
                image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

                content.append(sampler._handle_image(image_base64))

            prompt_messages = [
                sampler._pack_message(content=content, role="user")
            ]

            # If sampler is a SamplerBaseWithId, we need to pass the id to the __call__ method
            if isinstance(sampler, SamplerBaseWithId):
                response_text = sampler(prompt_messages, row["id"])
            else:
                response_text = sampler(prompt_messages)

            extracted_answer = answer_clean_with_parser(self.answer_parser, response_text)
            score = compare_answer_with_groundtruth(extracted_answer, row["Answer"])

            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["Answer"],
                extracted_answer=extracted_answer,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]

            single_eval_result = SingleEvalResult(html=html, score=score, convo=convo)

            # If sampler is a SamplerBaseWithId, we need to return a SingleResult
            if isinstance(sampler, SamplerBaseWithId):
                single_result = SingleResult(
                    task=self.name,
                    id=row["id"],
                    problem=SingleProblem(instruction=THEOREMQA_DEFAULT_PROMPT, input=row["Question"], target=row["Answer"]),
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
    