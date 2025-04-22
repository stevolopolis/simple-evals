"""
BBEH: BIG-Bench Extra Hard
Mehran Kazemi, Bahare Fatemi, Hritik Bansal, John Palowitch, Chrysovalantis Anastasiou
https://arxiv.org/abs/2502.19187
"""

import random
import requests

from .bbeh_eval import BBEHEval, BBEH_SUFFIX


BBH_URL = "https://github.com/suzgunmirac/BIG-Bench-Hard/raw/refs/heads/main/bbh"


class BBHEval(BBEHEval):
    def __init__(
        self,
        subtask: str,
        n_repeats: int = 4,
        num_examples: int | None = None,  # restrict to a subset of the data for debugging
        split_ratio: float | None = None  # split the data into training and evaluation sets
    ):
        if subtask is None:
            raise ValueError(f"Subtask must be provided. Available subtasks are:\n{self.subtasks}")
    
        assert subtask in self.subtasks, f"Provided subtask is not valid. Available subtasks are:\n{self.subtasks}"
        
        self.task_name = f"bbh-{subtask}"
        
        # Download task json from url
        task_url = f"{BBH_URL}/{subtask}.json"
        raw_data = requests.get(task_url).json()

        # Parse "examples" field for list of data
        examples = raw_data["examples"]

        # Add id column
        examples = [example | {"id": i} for i, example in enumerate(examples)]

        # Append prompt suffix to each example
        examples = [example | {"input": example["input"] + BBEH_SUFFIX} for example in examples]

        print(f"BBH: {self.task_name} ({len(examples)} examples)")

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

        self.examples = examples
        self.n_repeats = n_repeats

    @property
    def name(self):
        return "bbh"
    
    @property
    def subtasks(self):
        return [
            'boolean_expressions',
            'causal_judgement',
            'date_understanding',
            'disambiguation_qa',
            'dyck_languages',
            'formal_fallacies',
            'geometric_shapes',
            'hyperbaton',
            'logical_deduction_five_objects',
            'logical_deduction_seven_objects',
            'logical_deduction_three_objects',
            'movie_recommendation',
            'multistep_arithmetic_two',
            'navigate',
            'object_counting',
            'penguins_in_a_table',
            'reasoning_about_colored_objects',
            'ruin_names',
            'salient_translation_error_detection',
            'snarks',
            'sports_understanding',
            'temporal_sequences',
            'tracking_shuffled_objects_five_objects',
            'tracking_shuffled_objects_seven_objects',
            'tracking_shuffled_objects_three_objects',
            'web_of_lies',
            'word_sorting'
        ]   
    
    