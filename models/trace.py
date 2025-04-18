from typing import List, Union, Dict
from dataclasses import dataclass


@dataclass
class TraceNode:
    id: int
    input: Union[List[dict], None] = None
    output: Union[str, None] = None
    parent_node_ids: Union[int, None] = None


@dataclass
class Trace:
    nodes: Dict[int, TraceNode]
    n_nodes: int

    def __init__(self):
        self.nodes = {0: TraceNode(id=0)}
        self.n_nodes = 1

    def add_messages(self, input: List[dict], output: str, parent_node_ids: Union[int, List[int]] = -1):
        node_id = self.n_nodes
        # if parent_id is not specified, use the last node as parent
        if parent_node_ids == -1:
            parent_node_ids = 0

        if isinstance(parent_node_ids, int):
            parent_node_ids = [parent_node_ids]

        self.nodes[node_id] = TraceNode(id=node_id, input=input, output=output, parent_node_ids=parent_node_ids)
        self.n_nodes += 1

        return node_id
    
    def __dict__(self):
        return {
            "nodes": {node.id: node.__dict__ for node in self.nodes.values()},
            "n_nodes": self.n_nodes
        }
    

@dataclass
class Traces:
    traces: Dict[int, Trace]

    def __init__(self):
        self.traces = {}

    def __dict__(self):
        return {id: trace.__dict__() for id, trace in self.traces.items()}
    
    def __getitem__(self, id: int):
        return self.traces[id]
    
    def __setitem__(self, id: int, trace: Trace):
        self.traces[id] = trace