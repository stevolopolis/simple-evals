import copy
from typing import List, Union
from .base import BaseModel 

# PoT imports
from .src_impl.pot import safe_execute_subprocess


class LiteModel(BaseModel):
    @property
    def model_type(self):
        return "Vanilla"
    
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


class CoTModel(BaseModel):
    @property
    def model_type(self):
        return "CoT"

    def run(self, messages, id: int, return_node_id: bool = False, parent_node_ids: Union[int, List[int]] = -1):
        messages = copy.deepcopy(messages)
        # insert system prompt  
        messages.insert(0, self.system_prompt)
        # append cot prompt to final message
        cot_prompt = "\nFirst, think carefully step by step and then answer the question."
        messages[-1]['content'] += cot_prompt

        # first call to elicit cot
        thoughts, cot_node_id = self.completion(
            model=self.model_id,
            messages=messages,
            id=id,
            return_node_id=True,
            parent_node_ids=parent_node_ids
        )

        # append cot thoughts
        messages.append({
            "role": "assistant",
            "content": thoughts
        })

        # append final answer prompt
        messages.append({
            "role": "user",
            "content": f"So, what is the final answer? Answer in the specified format."
        })
        
        # second call to get final answer
        return self.completion(
            model=self.model_id,
            messages=messages,
            id=id,
            parent_node_ids=cot_node_id,
            return_node_id=return_node_id
        )
    

class CoTSCModel(CoTModel):
    @property
    def model_type(self):
        return "CoTSC"

    def run(self, messages, id: int, return_node_id: bool = False, parent_node_ids: Union[int, List[int]] = -1):
        samples = 40
        messages = copy.deepcopy(messages)

        # Sample CoT traces
        candidate_messages = [copy.deepcopy(messages) for _ in range(samples)]
        candidate_answers = []
        candidate_node_ids = []
        for sample in range(samples):
            sampled_messages = candidate_messages[sample]
            res, node_id = super().run(sampled_messages, id, return_node_id=True, parent_node_ids=parent_node_ids)
            candidate_answers.append(res)
            candidate_node_ids.append(node_id)

        # Select answer 
        return self.select_answer_direct(messages, id, candidate_answers, return_node_id=return_node_id, parent_node_ids=candidate_node_ids)
    
    def select_answer(self, messages, id: int, candidate_answers: List[str], return_node_id: bool = False, parent_node_ids: Union[int, List[int]] = -1):
        messages = copy.deepcopy(messages)
        # Insert candidates
        CANDIDATES_PROMPT = """
        I have underwent multiple thought processes and generated multiple answers. The answers are as follows:
        """.strip()

        candidates_message = CANDIDATES_PROMPT + "\n"
        for i, candidate in enumerate(candidate_answers):
            candidates_message += f"Answer {i+1}:\n{candidate}\n\n"

        messages.append(self._pack_message("assistant", candidates_message))

        # Insert sc prompt
        SC_PROMPT = """
        Given all the candidate answers, what is your final answer that agrees with the majority of the candidate answers?
        Remember to give your answer in the specified format, if it is provided.
        """.strip()
        messages.append(self._pack_message("user", SC_PROMPT))

        # Select majority vote answer with CoT
        return super().run(messages, id, parent_node_ids=parent_node_ids, return_node_id=return_node_id)

    def select_answer_direct(self, messages, id: int, candidate_answers: List[str], return_node_id: bool = False, parent_node_ids: Union[int, List[int]] = -1):
        """
        Simply get the majority vote answer with string exact matching.
        If there are multiple majority answers, return the first one.
        """
        messages = copy.deepcopy(messages)
        
        answers = {}
        for answer in candidate_answers:
            if answer not in answers:
                answers[answer] = 0
            answers[answer] += 1

        majority_answer = max(answers, key=answers.get)

        return majority_answer
    
    def select_answer_direct_with_permutations(self, messages, id: int, candidate_answers: List[str], return_node_id: bool = False, parent_node_ids: Union[int, List[int]] = -1):
        """
        
        """
        pass        


class LTMModel(BaseModel):
    @property
    def model_type(self):
        return "CoT"

    def run(self, messages, id: int, return_node_id: bool = False, parent_node_ids: Union[int, List[int]] = -1):
        messages = copy.deepcopy(messages)
        # insert system prompt  
        messages.insert(0, self.system_prompt)
        # append cot prompt to final message
        cot_prompt = "\Let's break down the problem into smaller sub-problems and solve each sub-problem one by one."
        messages[-1]['content'] += cot_prompt

        # first call to elicit cot
        thoughts, cot_node_id = self.completion(
            model=self.model_id,
            messages=messages,
            id=id,
            return_node_id=True,
            parent_node_ids=parent_node_ids
        )

        # append cot thoughts
        messages.append({
            "role": "assistant",
            "content": thoughts
        })

        # append final answer prompt
        messages.append({
            "role": "user",
            "content": f"So, what is the final answer? Answer in the specified format."
        })
        
        # second call to get final answer
        return self.completion(
            model=self.model_id,
            messages=messages,
            id=id,
            parent_node_ids=cot_node_id,
            return_node_id=return_node_id
        )


class PoTModel(BaseModel):
    @property
    def model_type(self):
        return "PoT"
    
    def run(self, messages, id: int, return_node_id: bool = False, parent_node_ids: Union[int, List[int]] = -1):
        messages = copy.deepcopy(messages)
        messages.insert(0, self.system_prompt)
        
        messages[-1]['content'] += """
        To solve the problem, you need to implement a solver() function in Python that solves the problem by returning the answer.
        Make sure the function adheres to the Python syntax.
        Do not right any code outside the solver() function.
        The function should not require any external libraries besides math and numpy.
        Do not include any other text after the code block.

        For example, if the problem is to find the sum of the first 100 natural numbers, you should write:
        ```python
        import math
        import numpy as np
        
        def solver():
            return sum(range(1, 101))
        ```

        Write your code in the following format:
        ```python
        import math
        import numpy as np
        
        def solver():
            # Let's think step by step to derive the answer, and then return the answer
            # In the first step, we can define the following variable:
            <your code here>
            return answer

        Now, given the question, what is the solver() function?
        """
        
        code_res, node_id = self.completion(
            model=self.model_id,
            messages=messages,
            id=id,
            return_node_id=True,
            parent_node_ids=parent_node_ids,
            logit_bias={"1303": -2}     # suppress "#" token to encourage it to generate code
        )
        
        # Parse the code from the response
        code = code_res.split("```python")[1].split("```")[0]
        # Get return value of the function
        code += "\nans = solver()"
        code += '\nprint(f"Answer: {ans}")'

        # Execute the code
        ans = safe_execute_subprocess(code)
        
        # Query the model again with the executed result to general the final answer
        messages.append({
            "role": "assistant",
            "content": code_res
        })

        # append final answer prompt
        messages.append({
            "role": "user",
            "content": f"The result of your program is {ans}. Please give your final answer in the specified format, if it is provided."
        })
        
        # second call to get final answer
        return self.completion(
            model=self.model_id,
            messages=messages,
            id=id,
            parent_node_ids=node_id,
            return_node_id=return_node_id
        )


class SelfRefineModel(BaseModel):
    @property
    def model_type(self):
        return "Self-refine"
    
    def run(self, messages, id: int, return_node_id: bool = False, parent_node_ids: Union[int, List[int]] = -1):
        messages = copy.deepcopy(messages)
        messages.insert(0, self.system_prompt)

        # get initial solution 
        solution, prev_node_id = self.init_solution(messages, id, parent_node_ids)

        # iterative refinement
        n_refines = 3
        for i in range(n_refines):
            fb, solution, prev_node_id = self.iterative_refine(messages, solution, id, prev_node_id)

            if "the solution is correct" in fb.lower():
                break

        # Refine answer with specified format
        messages.append({
            "role": "assistant",
            "content": solution
        })
        # append final answer prompt
        messages.append({
            "role": "user",
            "content": f"You've found the final asnwer. Now answer in the specified format."
        })
        
        # second call to get final answer
        return self.completion(
            model=self.model_id,
            messages=messages,
            id=id,
            parent_node_ids=prev_node_id,
            return_node_id=return_node_id
        )        

    def init_solution(self, messages, id: int, parent_node_ids: Union[int, List[int]] = -1):
        messages = copy.deepcopy(messages)

        # Append custom answer format
        messages[-1]["content"] += """
        In this step, Give your answer in the format of 'ANSWER: <your answer>'. Do not include any other text after the answer.
        """.strip()

        init_sol, prev_node_id = self.completion(
            model=self.model_id,
            messages=messages,
            id=id,
            parent_node_ids=parent_node_ids,
            return_node_id=True
        )

        # Parse the solution from the response
        split_sol = init_sol.split("Answer:")
        if len(split_sol) > 1:
            solution = split_sol[1].strip()
        else:
            solution = init_sol

        return solution, prev_node_id

    def iterative_refine(self, messages, solution: str, id: int, parent_node_ids: Union[int, List[int]] = -1):
        fb_prompt = """
In this step, you need to carefully verify the correctness of the previous thoughts with natural
language.
If it is correct, explicitly say that "the solution is correct" and repeat the solution with the following format: "Answer:\n<solution>".
If it is incorrect, state the reason of error and go through the solution step by step to derive the correct answer. Then, give your refined answer in the the following format: "Answer:\n<solution>".
        """.strip()

        fb_messages = copy.deepcopy(messages)

        fb_messages.append({
            "role": "assistant",
            "content": solution
        })
        fb_messages.append({
            "role": "user",
            "content": fb_prompt
        })

        fb_and_solution, prev_node_id = self.completion(
            model=self.model_id,
            messages=fb_messages,
            id=id,
            parent_node_ids=parent_node_ids,
            return_node_id=True
        )

        fb = fb_and_solution.split("Answer:")[0]
        refined_solution = fb_and_solution.split("Answer:")[1]

        return fb, refined_solution, prev_node_id