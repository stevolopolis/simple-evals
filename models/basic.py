from .base import BaseModel 
import copy

class LiteModel(BaseModel):
    @property
    def model_type(self):
        return "Vanilla"
    
    def run(self, messages, id: int):
        messages.insert(0, self.system_prompt)
        res = self.completion(
            model=self.model_id,
            messages=messages,
            id=id
        )
        return res


class CoTModel(BaseModel):
    @property
    def model_type(self):
        return "CoT"

    def run(self, messages, id: int):
        messages.insert(0, self.system_prompt)
        # append cot prompt to final message
        cot_prompt = "\nFirst, think carefully step by step and then answer the question."
        messages[-1]['content'] += cot_prompt

        # first call to elicit cot
        thoughts = self.completion(
            model=self.model_id,
            messages=messages,
            id=id
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
        res = self.completion(
            model=self.model_id,
            messages=messages,
            id=id
        )
        return res
    
