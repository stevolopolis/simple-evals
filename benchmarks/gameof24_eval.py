"""
Measuring Mathematical Problem Solving With the MATH Dataset
Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, Jacob Steinhardt
https://arxiv.org/abs/2103.03874
"""

import random
import re
import sympy
from typing import List, Tuple, Union

import pandas

from . import common
from .common import HTML_JINJA
from ..types import Eval, EvalResult, SamplerBase, SamplerBaseWithId, SingleEvalResult, SingleResult, SingleProblem

BOT_PROMPT = """
Let's play a game called 24. You'll be given four integers, and your objective is to use each number only once, combined with any of the four arithmetic operations (addition, subtraction, multiplication, and division) and parentheses, to achieve a total of 24. For example, if the input is 4, 7, 8, and 8, the output could be 7 * 8 - 4 * 8 = 24. You only need to find one feasible solution!
Input: {input}

{answer_pattern}
""".strip()


# Default from original ToT code
ORIGINAL_PROMPT = """
standard_prompt = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24.
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) = 24
Input: 2 9 10 12
Answer: 2 * 12 * (10 - 9) = 24
Input: 4 9 10 13
Answer: (13 - 9) * (10 - 4) = 24
Input: 1 4 8 8
Answer: (8 / 4 + 1) * 8 = 24
Input: 5 5 5 9
Answer: 5 + 5 + 5 + 9 = 24
Input: {input}

{answer_pattern}
""".strip()

QUERY_TEMPLATE = BOT_PROMPT


class Gameof24Eval(Eval):
    def __init__(
        self,
        num_examples: int | None = None,
        n_repeats: int = 16,
        split_ratio: float | None = None  # split the data into training and evaluation sets
    ):
        super().__init__()
        # Load data
        data_path = f'simple-evals/benchmarks/data/gameof24.jsonl'
        df = pandas.read_json(data_path, lines=True)    # header: {input, target}

        # Create index column
        df['id'] = df.index

        # Init answer pattern
        self.answer_pattern = common.DefaultParser()

        # Add answer pattern column
        df['answer_pattern'] = self.answer_pattern.answer_pattern

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

    @property
    def name(self):
        return "gameof24"

    def __call__(self, sampler: Union[SamplerBase, SamplerBaseWithId]) -> Union[EvalResult, Tuple[EvalResult, List[SingleEvalResult]]]:
        def fn(row: dict):
            input_, target = self.get_x_y_data(row)
            prompt_messages = [
                sampler._pack_message(content=input_, role="user")
            ]
            # If sampler is a SamplerBaseWithId, we need to pass the id to the __call__ method
            if isinstance(sampler, SamplerBaseWithId):
                response_text = sampler(prompt_messages, row["id"])
            else:
                response_text = sampler(prompt_messages)

            extracted_answer = self.answer_pattern.parse(response_text)
            score = float(self.check_equality(row["input"], extracted_answer))
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=target,
                extracted_answer=extracted_answer,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]

            single_eval_result = SingleEvalResult(html=html, score=score, convo=convo)

            # If sampler is a SamplerBaseWithId, we need to return a SingleResult
            if isinstance(sampler, SamplerBaseWithId):
                single_result = SingleResult(
                    task=self.name,
                    id=row["id"],
                    problem=SingleProblem(instruction=QUERY_TEMPLATE, input=row["input"], target=row["target"]),
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

    def check_equality(self, input, output):
        """
        Given an input and output, check if the output is correct and follows the rules of the game.

        Taken from original ToT code: https://github.com/princeton-nlp/tree-of-thought-llm/blob/master/src/tot/tasks/game24.py#L44
        """
        expression = output.strip().split('\n')[-1].lower().replace('answer: ', '').split('=')[0]
        numbers = re.findall(r'\d+', expression)
        problem_numbers = re.findall(r'\d+', input)
        if sorted(numbers) != sorted(problem_numbers):
            return 0
        try:
            # print(sympy.simplify(expression))
            return int(sympy.simplify(expression) == 24)
        except Exception as e:
            # print(e)
            return 0
        
    def eval_fn(self, sample, reference):
        extracted_answer = self.answer_pattern.parse(sample)
        return self.check_equality(reference, extracted_answer)   
        
    def get_x_y_data(self, example):
        x = QUERY_TEMPLATE.format(**example)
        y = example["input"]
        return x, y