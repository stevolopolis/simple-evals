import copy
from typing import List, Union
from .base import BaseModel 


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

