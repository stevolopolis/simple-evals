from dataclasses import dataclass, field
from typing import Any

from torch.utils.data import DataLoader

Message = dict[str, Any]  # keys role, content
MessageList = list[Message]


class SamplerBase:
    """
    Base class for defining a sampling model, which can be evaluated,
    or used as part of the grading process.
    """

    def __call__(self, message_list: MessageList) -> str:
        raise NotImplementedError


class SamplerBaseWithId(SamplerBase):
    """
    SamplerBase wrapper that adds an id parameter to the __call__ method.
    Used for custom models with traces.
    """

    def __call__(self, message_list: MessageList, id: int) -> str:
        raise NotImplementedError



@dataclass
class EvalResult:
    """
    Result of running an evaluation (usually consisting of many samples)
    """

    score: float | None  # top-line metric
    metrics: dict[str, float] | None  # other metrics
    htmls: list[str]  # strings of valid HTML
    convos: list[MessageList]  # sampled conversations


@dataclass
class SingleEvalResult:
    """
    Result of evaluating a single sample
    """

    score: float | None
    metrics: dict[str, float] = field(default_factory=dict)
    html: str | None = None
    convo: MessageList | None = None  # sampled conversation


@dataclass
class SingleProblem:
    """
    A single problem. For logging purposes only.
    """
    instruction: str | None
    input: str | None
    target: str | None
    

@dataclass 
class SingleResult:
    """
    Result of evaluating a single sample. For logging purposes only.
    """
    task: str
    id: int
    problem: SingleProblem
    output: Any
    answer: Any
    score: float | None


class Eval:
    """
    Base class for defining an evaluation.
    """
    def __call__(self, sampler: SamplerBase) -> EvalResult:
        raise NotImplementedError
    
    def get_data(self):
        return self.examples
    
    def eval_fn(self, sample, reference, **kwargs):
        raise NotImplementedError
    
    def get_x_y_data(self, example):
        raise NotImplementedError
    
    def get_train_data_loader(self, batch_size: int = 16, shuffle: bool = True):
        # parse examples into simple (x, y) pairs
        training_data = []
        for example in self.training_examples:
            training_data.append(self.get_x_y_data(example))

        return DataLoader(training_data, batch_size=batch_size, shuffle=shuffle)
    
    def get_eval_data_loader(self, batch_size: int = 16, shuffle: bool = True):
        # parse examples into simple (x, y) pairs
        eval_data = []
        for example in self.examples:
            eval_data.append(self.get_x_y_data(example))

        return DataLoader(eval_data, batch_size=batch_size, shuffle=shuffle)
