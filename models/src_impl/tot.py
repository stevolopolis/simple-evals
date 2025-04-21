import itertools
import numpy as np


class ToTModel(BaseModel):
    """
    All helper functions are copied from the original ToT implementation, with some modifications to fit the current codebase.
    Modifications include:
        - replace the `gpt` function with a wrapper of the `completion` function.
        - replace the `task` class with prompt.
    """
    def __post__init__(self):
        self.steps = []

    def set_steps(self, steps: str):
        self.steps = steps

    @property
    def model_type(self):
        return "ToT"
    
    def run(self, messages, id: int, return_node_id: bool = False, parent_node_ids: Union[int, List[int]] = -1):
        messages = copy.deepcopy(messages)
        messages.insert(0, self.system_prompt)

        method_generate = 'sample'
        method_evaluate = 'vote'
        method_select = 'greedy'
        n_generate_sample = 10
        n_evaluate_sample = 10
        n_select_sample = 1     
        
        
        x = None
        ys = ['']  # current output candidates
        subproblems = []
        subanswers = []
        curr_subproblem = None
        for step in range(steps):
            if step == 0:
                messages[-1]["content"] += f"""
                First, let's break the problem down into smaller sub-problems. Answer with the following format:
                sub-problem 1: ...
                sub-problem 2: ...
                ...
                # of sub-problems: ...
                """.strip()
            else:
                messages[-1]["content"] += f"""
                To solve the problem, we need to solve the following sub-problems:
                {'\n'.join(subproblems)}
                """.strip()

                messages[-1]["content"] += f"""
                You have already solved the following sub-problems and their solutions are as follows:
                {'\n'.join(subanswers)}
                """.strip()

                messages[-1]["content"] += f"""
                Now, let's solve the following sub-problem:
                {curr_subproblem}
                """.strip()

            # generation
            if method_generate == 'sample':
                new_ys = [get_samples(messages, y, n_generate_sample) for y in ys]
            elif method_generate == 'propose':
                k = 5
                messages[-1]["content"] = f"Give me {k} different solutions to the problem."
                new_ys = [get_proposals(messages, y) for y in ys]

            new_ys = list(itertools.chain(*new_ys))
            ids = list(range(len(new_ys)))

            # evaluation
            if method_evaluate == 'vote':
                values = get_votes(messages, new_ys, n_evaluate_sample)
            elif method_evaluate == 'value':
                values = get_values(messages, new_ys, n_evaluate_sample)

            # selection
            if method_select == 'sample':
                ps = np.array(values) / sum(values)
                select_ids = np.random.choice(ids, size=n_select_sample, p=ps).tolist()
            elif method_select == 'greedy':
                select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:n_select_sample]

            select_new_ys = [new_ys[select_id] for select_id in select_ids]
            ys = select_new_ys

        return ys
