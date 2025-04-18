from typing import Any
import json
import copy
from litellm import completion
from typing import Union, List

from ..types import SamplerBaseWithId
from .trace import Trace, Traces


SYSTEM_PROMPT = """
You are a helpful assistant.
"""


class BaseModel(SamplerBaseWithId):
    def __init__(self, model_id):
        self.model_id = model_id
        self.system_prompt = {
            "role": "system",
            "content": SYSTEM_PROMPT
        }

        self.traces = {}
        self.full_traces = Traces()   # contains every I/O call of the LLM

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
    
    def completion(self,
        model: str,
        messages: list[dict],
        simple: bool = True,
        id: int = None,
        parent_node_ids: Union[int, List[int]] = -1,
        return_node_id: bool = False,
        **kwargs
    ):
        """
        Wraps the litellm completion function to support trace logging.

        Args:
            messages: list of messages
            simple: if True, return the first choice's message content
        """
        res = completion(
            model=model,
            messages=messages,
            cache={"no-cache": True},
            **kwargs      
        )

        # add previous messages to trace, if not already added
        self.traces[id].extend(copy.deepcopy(messages[len(self.traces[id]):]))
        # add I/O to full trace
        curr_node_id = self.full_traces[id].add_messages(
            input=copy.deepcopy(messages),
            output=copy.deepcopy(res['choices'][0]['message']['content']),
            parent_node_ids=parent_node_ids
        )

        if not return_node_id:
            if simple:
                return res['choices'][0]['message']['content']
            else:
                return res['choices']
        else:
            if simple:
                return res['choices'][0]['message']['content'], curr_node_id
            else:
                return res['choices'], curr_node_id

    def save_trace(self, path: str):
        # save traces
        with open(path, 'w') as f:
            f.write(json.dumps(self.traces, indent=4))
        
        # save full traces
        with open(path.replace('.json', '_full.json'), 'w') as f:
            f.write(json.dumps(self.full_traces.__dict__(), indent=4))
        

    def run(self, messages, id: int, return_node_id: bool = False, parent_node_ids: Union[int, List[int]] = -1):
        raise NotImplementedError("The run method must be implemented by the subclass.")
    
    def __call__(self, messages, id: int):
        # copy messages to avoid modifying the original messages
        messages = copy.deepcopy(messages)
        
        # Check if id is already in traces. If yes, it means we are re-running with a different seed.
        # In this case, we will append the seed number to the id.
        if id in self.traces:
            id = f"{id}_{len(self.traces[id]) + 1}"

        self.traces[id] = messages

        self.full_traces[id] = Trace()
        res = self.run(messages, id)

        # add final response to trace
        self.traces[id].append({
            "role": "assistant",
            "content": res
        })


        return res