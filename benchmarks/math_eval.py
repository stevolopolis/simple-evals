"""
Measuring Mathematical Problem Solving With the MATH Dataset
Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, Jacob Steinhardt
https://arxiv.org/abs/2103.03874
"""

import copy
import random
import re
from typing import Literal, List, Tuple, Union

import pandas

from . import common
from .common import ANSWER_PATTERN, HTML_JINJA, check_equality
from ..types import Eval, EvalResult, SamplerBase, SamplerBaseWithId, SingleEvalResult, SingleResult, SingleProblem

# Default in simple-evals
QUERY_TEMPLATE_SIMPLE_EVAL = """
Solve the following math problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.

{Question}

Remember to put your answer on its own line after "Answer:", and you do not need to use a \\boxed command.
""".strip()

QUERY_TEMPLATE_DOTS = """
Solve the following math problem. The last line of your response should be of the form 'Answer: \\boxed{$ANSWER}' (without quotes) where $ANSWER is the answer to the problem. If the answer is a fraction, do not convert it to a decimal.

Question: {Question}
""".strip()

QUERY_TEMPLATE = QUERY_TEMPLATE_DOTS


class MathEval(Eval):
    def __init__(
        self,
        equality_checker: SamplerBase,
        num_examples: int | None = None,
        n_repeats: int = 16,
        split: Literal["math_test", "math_500_test"] = "math_500_test",
        split_ratio: float | None = None  # split the data into training and evaluation sets
    ):
        super().__init__()

        df = pandas.read_csv(
            f"https://openaipublic.blob.core.windows.net/simple-evals/{split}.csv"
        )
        
        # Create index column
        df['id'] = df.index

        examples = [row.to_dict() for _, row in df.iterrows()]

        if split_ratio:
            split_index = int(len(examples) * split_ratio)
            # shuffle examples
            random.shuffle(examples)
            self.training_examples = examples[:split_index]
            examples = examples[split_index:]
        else:
            self.training_examples = examples

        if num_examples:
            assert n_repeats == 1, "n_repeats only supported for num_examples = None"
            rng = random.Random(0)
            examples = rng.sample(examples, num_examples)
        self.examples = examples * n_repeats
        self.equality_checker = equality_checker

    @property
    def name(self):
        return "math"

    def __call__(self, sampler: Union[SamplerBase, SamplerBaseWithId]) -> Union[EvalResult, Tuple[EvalResult, List[SingleEvalResult]]]:
        def fn(row: dict):
            input_, target_ = self.get_x_y_data(row)

            prompt_messages = [
                sampler._pack_message(content=input_, role="user")
            ]
            # If sampler is a SamplerBaseWithId, we need to pass the id to the __call__ method
            if isinstance(sampler, SamplerBaseWithId):
                response_text = sampler(prompt_messages, row["id"])
            else:
                response_text = sampler(prompt_messages)

            match = re.search(ANSWER_PATTERN, response_text)
            extracted_answer = match.group(1) if match else None
            score = float(check_equality(self.equality_checker, target_, extracted_answer))
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=target_,
                extracted_answer=extracted_answer,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]

            single_eval_result = SingleEvalResult(html=html, score=score, convo=convo)

            # If sampler is a SamplerBaseWithId, we need to return a SingleResult
            if isinstance(sampler, SamplerBaseWithId):
                single_result = SingleResult(
                    task=self.name,
                    id=row["id"],
                    problem=SingleProblem(instruction=QUERY_TEMPLATE, input=row["Question"], target=row["Answer"]),
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
        match = re.search(ANSWER_PATTERN, sample)
        extracted_answer = match.group(1) if match else None
        score = float(check_equality(self.equality_checker, reference, extracted_answer))
        if return_extracted_answer:
            return score, extracted_answer
        return score
    
    def get_x_y_data(self, example):
        x = QUERY_TEMPLATE.replace("{Question}", example["Question"])
        y = example["Answer"]
        return x, y
