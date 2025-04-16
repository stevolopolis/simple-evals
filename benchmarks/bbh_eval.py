"""
BBEH: BIG-Bench Extra Hard
Mehran Kazemi, Bahare Fatemi, Hritik Bansal, John Palowitch, Chrysovalantis Anastasiou
https://arxiv.org/abs/2502.19187
"""

import random
import requests

from .bbeh_eval import BBEHEval


BBH_URL = "https://github.com/suzgunmirac/BIG-Bench-Hard/raw/refs/heads/main/bbh"


class BBHEval(BBEHEval):
    def __init__(
        self,
        n_repeats: int = 4,
        num_examples: int | None = None,  # restrict to a subset of the data for debugging
        subtask: str | None = None,
    ):
        if subtask is None:
            raise ValueError(f"Subtask must be provided. Available subtasks are:\n{self.subtasks}")
    
        assert subtask in self.subtasks, f"Provided subtask is not valid. Available subtasks are:\n{self.subtasks}"
        
        task_name = subtask
        
        # Download task json from url
        task_url = f"{BBH_URL}/{task_name}.json"
        raw_data = requests.get(task_url).json()

        # Parse "examples" field for list of data
        examples = raw_data["examples"]

        # Add id column
        examples = [example | {"id": i} for i, example in enumerate(examples)]

        print(f"BBH: {task_name} ({len(examples)} examples)")

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
    