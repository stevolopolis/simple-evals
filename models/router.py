from ..models import *


model_methods = {
    "cot": CoTModel,
    "vanilla": LiteModel,
    "dspy": DspyModel,
    "dspy-cot": DspyCoTModel,
    "dspy-pot": DspyPotModel
}



def get_model_from_id(model_id: str, method: str) -> BaseModel:
    if method not in model_methods:
        raise ValueError(f"Invalid method: {method}")
    
    return model_methods[method](model_id)
    

def get_all_methods():
    return model_methods.keys()