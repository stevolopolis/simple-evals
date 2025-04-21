from .base import BaseModel

import dspy


class DspyModel(BaseModel):
    @property
    def model_type(self):
        return "Dspy"
    
    def run(self, messages, id: int):
        lm = dspy.LM(self.model_id)

        with dspy.context(lm=lm):
            module = dspy.Predict("question -> answer")
            question = messages[-1]['content']
            answer = module(question=question)
            
        return answer.answer



class DspyCoTModel(BaseModel):
    @property
    def model_type(self):
        return "CoT-Dspy"
    
    def run(self, messages, id: int):
        lm = dspy.LM(self.model_id)

        with dspy.context(lm=lm):
            module = dspy.ChainOfThought("question -> answer")
            question = messages[-1]['content']
            try:
                answer = module(question=question)
            # Sometimes the model reasons over the token limit, so no answer is returned.
            # This causes an error within dspy, so we catch the error and return the error message.
            except RuntimeError as e:
                print(f"Error: {e}")
                return f"Error: {e}"
                

        return answer.answer
    

class DspyPotModel(BaseModel):
    @property
    def model_type(self):
        return "Pot-Dspy"
    
    def run(self, messages, id: int):
        lm = dspy.LM(self.model_id)

        with dspy.context(lm=lm):
            module = dspy.ProgramOfThought("question -> answer")
            question = messages[-1]['content']
            try:
                answer = module(question=question)
            except RuntimeError as e:
                print(f"Error: {e}")
                return f"Error: {e}"
            
        return answer.answer
