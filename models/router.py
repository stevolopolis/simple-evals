from ..models import *


model_methods = {
    "vanilla": LiteModel,
    "cot": CoTModel,
    "cot-sc": CoTSCModel,
    "ltm": LTMModel,
    "pot": PoTModel,
    "sr": SelfRefineModel,
    "dspy": DspyModel,
    "dspy-cot": DspyCoTModel,
    "dspy-pot": DspyPotModel,
    "tg": TGModel,
    "meta": MetaModel
}



def get_model_from_id(model_id: str, method: str) -> BaseModel:
    if method not in model_methods:
        raise ValueError(f"Invalid method: {method}")
    
    return model_methods[method](model_id)
    

def get_all_methods():
    return model_methods.keys()