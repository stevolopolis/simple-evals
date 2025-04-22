"""
Measuring Massive Multitask Language Understanding
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, Jacob Steinhardt
https://arxiv.org/abs/2009.03300
"""

import random
import re

import pandas

from . import common
from .common import (
    HTML_JINJA,
    MULTILINGUAL_ANSWER_PATTERN_TEMPLATE,
    MULTILINGUAL_ANSWER_REGEXES,
    format_multichoice_question,
    normalize_extracted_answer,
    normalize_response,
)
from ..types import Eval, EvalResult, SamplerBase, SamplerBaseWithId, SingleEvalResult, SingleResult, SingleProblem

subject2category = {
    "abstract_algebra": "stem",
    "anatomy": "other",
    "astronomy": "stem",
    "business_ethics": "other",
    "clinical_knowledge": "other",
    "college_biology": "stem",
    "college_chemistry": "stem",
    "college_computer_science": "stem",
    "college_mathematics": "stem",
    "college_medicine": "other",
    "college_physics": "stem",
    "computer_security": "stem",
    "conceptual_physics": "stem",
    "econometrics": "social_sciences",
    "electrical_engineering": "stem",
    "elementary_mathematics": "stem",
    "formal_logic": "humanities",
    "global_facts": "other",
    "high_school_biology": "stem",
    "high_school_chemistry": "stem",
    "high_school_computer_science": "stem",
    "high_school_european_history": "humanities",
    "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_mathematics": "stem",
    "high_school_microeconomics": "social_sciences",
    "high_school_physics": "stem",
    "high_school_psychology": "social_sciences",
    "high_school_statistics": "stem",
    "high_school_us_history": "humanities",
    "high_school_world_history": "humanities",
    "human_aging": "other",
    "human_sexuality": "social_sciences",
    "international_law": "humanities",
    "jurisprudence": "humanities",
    "logical_fallacies": "humanities",
    "machine_learning": "stem",
    "management": "other",
    "marketing": "other",
    "medical_genetics": "other",
    "miscellaneous": "other",
    "moral_disputes": "humanities",
    "moral_scenarios": "humanities",
    "nutrition": "other",
    "philosophy": "humanities",
    "prehistory": "humanities",
    "professional_accounting": "other",
    "professional_law": "humanities",
    "professional_medicine": "other",
    "professional_psychology": "social_sciences",
    "public_relations": "social_sciences",
    "security_studies": "social_sciences",
    "sociology": "social_sciences",
    "us_foreign_policy": "social_sciences",
    "virology": "other",
    "world_religions": "humanities",
}


class MMLUEval(Eval):
    def __init__(self, num_examples: int | None = None, language: str = "EN-US", split_ratio: float | None = None):
        super().__init__()
        if language != "EN-US":
            url = f"https://openaipublic.blob.core.windows.net/simple-evals/mmlu_{language}.csv"
        else:
            url = "https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv"
        df = pandas.read_csv(url)

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
            examples = random.Random(0).sample(examples, num_examples)
        self.examples = examples

    @property
    def name(self):
        return "mmlu"

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
            category = subject2category.get(row["Subject"], "other")
            
            single_eval_result = SingleEvalResult(html=html, score=score, convo=convo, metrics={category: score})

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

    def get_x_y_data(self, example):
        x = format_multichoice_question(example)
        y = example["Answer"]
        return x, y
    
    def eval_fn(self, sample, reference, return_extracted_answer=False):
        response_text = normalize_response(sample)
        extracted_answer = None
        for answer_regex in MULTILINGUAL_ANSWER_REGEXES:
            regex = MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(answer_regex)
            match = re.search(regex, response_text)
            if match:
                extracted_answer = normalize_extracted_answer(match.group(1))
                break
        score = 1.0 if extracted_answer == reference else 0.0
        if return_extracted_answer:
            return score, extracted_answer
        return score
