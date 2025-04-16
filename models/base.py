from .types import SamplerBase
from typing import Any

SYSTEM_PROMPT = """
You are a helpful assistant.
"""


class BaseModel(SamplerBase):
    def __init__(self, model_id):
        self.model_id = model_id
        self.system_prompt = {
            "role": "system",
            "content": SYSTEM_PROMPT
        }

        print(f"Initializing model.\nModel ID: {self.model_id}\nModel Type: {self.model_type}")

    @property
    def model_type(self):
        raise NotImplementedError("The model_type property must be implemented by the subclass.")

    def _handle_image(
        self, image: str, encoding: str = "base64", format: str = "png", fovea: int = 768
    ):
        """Copied from openai/simple-evals/sampler/chat_completion_sampler.py"""
        new_image = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{format};{encoding},{image}",
            },
        }
        return new_image

    def _handle_text(self, text: str):
        """Copied from openai/simple-evals/sampler/chat_completion_sampler.py"""
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        """Copied from openai/simple-evals/sampler/chat_completion_sampler.py"""
        return {"role": str(role), "content": content}

    def run(self, messages):
        raise NotImplementedError("The run method must be implemented by the subclass.")
    
    def __call__(self, messages):
        return self.run(messages)