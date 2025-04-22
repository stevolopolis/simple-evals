"""
GPQA: A Graduate-Level Google-Proof Q&A Benchmark
David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, Samuel R. Bowman
https://arxiv.org/abs/2311.12022
"""

import random
import re

import pandas

from . import common
from .common import ANSWER_PATTERN_MULTICHOICE, HTML_JINJA, format_multichoice_question
from ..types import Eval, EvalResult, MessageList, SamplerBase, SamplerBaseWithId, SingleEvalResult, SingleResult, SingleProblem


class GPQAEval(Eval):
    def __init__(
        self,
        n_repeats: int = 4,
        variant: str = "diamond",
        num_examples: int | None = None,  # restrict to a subset of the data for debugging
        split_ratio: float | None = None  # split the data into training and evaluation sets
    ):
        super().__init__()
        df = pandas.read_csv(
            f"https://openaipublic.blob.core.windows.net/simple-evals/gpqa_{variant}.csv"
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

        rng = random.Random(0)
        if num_examples:
            assert n_repeats == 1, "n_repeats only supported for num_examples = None"
            examples = rng.sample(examples, num_examples)
        examples = examples * n_repeats
        examples = [example | {"permutation": rng.sample(range(4), 4)} for example in examples]
        self.examples = examples
        self.n_repeats = n_repeats

    @property
    def name(self):
        return "gpqa"

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            input_, target = self.get_x_y_data(row)
        
            prompt_messages = [
                sampler._pack_message(
                    content=input_, role="user"
                )
            ]
            # If sampler is a SamplerBaseWithId, we need to pass the id to the __call__ method
            if isinstance(sampler, SamplerBaseWithId):
                response_text = sampler(prompt_messages, row["id"])
            else:
                response_text = sampler(prompt_messages)

            score, extracted_answer = self.eval_fn(response_text, target, return_extracted_answer=True)
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=target,
                extracted_answer=extracted_answer,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            
            single_eval_result = SingleEvalResult(
                html=html, score=score, convo=convo, metrics={"chars": len(response_text)}
            )

            # If sampler is a SamplerBaseWithId, we need to return a SingleResult
            if isinstance(sampler, SamplerBaseWithId):
                single_result = SingleResult(
                    task=self.name,
                    id=row["id"],
                    problem=SingleProblem(instruction=input_, input=row["Question"], target=target),
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
        match = re.search(ANSWER_PATTERN_MULTICHOICE, sample)
        extracted_answer = match.group(1) if match else None
        score = 1.0 if extracted_answer == reference else 0.0
        if return_extracted_answer:
            return score, extracted_answer
        return score
    
    def get_x_y_data(self, example):
        choices = [
            example["Correct Answer"],
            example["Incorrect Answer 1"],
            example["Incorrect Answer 2"],
            example["Incorrect Answer 3"],
        ]
        choices = [choices[i] for i in example["permutation"]]
        correct_index = choices.index(example["Correct Answer"])
        choices_dict = dict(
            A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=example["Question"]
        )

        x = format_multichoice_question(choices_dict)
        y = "ABCD"[correct_index]

        return x, y