from .base import BaseModel
from .src_impl.ape import *

from typing import Union, List
import copy
from tqdm import tqdm

import textgrad as tg
from textgrad.autograd.string_based_ops import StringBasedFunction
from ..types import Eval

COMPLETION_TYPE = "completion"      # {"completion", "chat"}

PROMPT_GEN_TEMPLATE = FORWARD_GEN_TEMPLATE
EVAL_TEMPLATE = ZERO_SHOT_EVAL_TEMPLATE

class TGModel(BaseModel):
    def __init__(self, model_id):
        super().__init__(model_id)

        self.llm_engine = tg.get_engine(f"experimental:{self.model_id}")
        tg.set_backward_engine(self.llm_engine, override=True)

    @property
    def model_type(self):
        return "TG"
    
    def run(self, messages, id: int, return_node_id: bool = False, parent_node_ids: Union[int, List[int]] = -1):
        messages = copy.deepcopy(messages)
        messages.insert(0, self.system_prompt)

        return self.completion(
            model=self.model_id,
            messages=messages,
            id=id,
            parent_node_ids=parent_node_ids,
            return_node_id=return_node_id
        )

    def optimize(self, benchmark: Eval, num_epochs: int = 3):
        # get helper funcs from benchmark
        eval_fn = benchmark.eval_fn
        dataloader = benchmark.get_train_data_loader()

        # eval fn
        def string_based_eval_fn(prediction, ground_truth_answer):
            return int(eval_fn(str(prediction.value), str(ground_truth_answer.value)))
        fn_purpose = "The runtime of string-based function that checks if the prediction is correct."
        tg_eval_fn = StringBasedFunction(string_based_eval_fn, function_purpose=fn_purpose)

        system_prompt = tg.Variable(self.system_prompt["content"],
                                    requires_grad=True,
                                    role_description="system prompt to guide the LLM's reasoning strategy for accurate responses")

        model = tg.BlackboxLLM(self.llm_engine, system_prompt=system_prompt)
        optimizer = tg.TGD(parameters=list(model.parameters()))
        
        # Get initial eval results  
        print("Getting initial eval results:")
        eval_results = benchmark(self)
        print(eval_results)

        print("Optimizing...")
        pbar = tqdm(total=num_epochs*len(dataloader))
        for epoch in range(num_epochs):
            for steps, (x_batch, y_batch) in enumerate(dataloader):
                losses = []
                for x, y in zip(x_batch, y_batch):
                    x = tg.Variable(x, role_description="query to the language model", requires_grad=False)
                    y = tg.Variable(y, role_description="correct answer to the query", requires_grad=False)

                    response = model(x)

                    loss = tg_eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
                    losses.append(loss)

                total_loss = tg.sum(losses)
                total_loss.backward()
                optimizer.step()

                pbar.update(1)

            # Update system prompt
            self.system_prompt["content"] = system_prompt.value

            # Run eval
            eval_results = benchmark(self)
            print(eval_results)

        # Update system prompt
        self.system_prompt["content"] = system_prompt.value