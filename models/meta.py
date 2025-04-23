from .base import BaseModel
from .src_impl.pot import safe_execute_subprocess
import copy
import re
import json

from typing import Any, Dict, List, Tuple, Union

class MetaModel(BaseModel):
    def __init__(self, model_id: str, cfg_path: str = "simple-evals/models/src_impl/meta_cfg.json") -> None:
        super().__init__(model_id)
        self._set_cfg_from_file(cfg_path)

    @property
    def model_type(self):
        return "meta"
    
    def run(self, messages, id: int, return_node_id: bool = False, parent_node_ids: Union[int, List[int]] = -1):
        parser_messages = copy.deepcopy(messages)

        # Insert meta prompt system prompt
        meta_messages = copy.deepcopy(self.meta_model_settings["message-list"])
        # Insert metaprompting prefix and suffix
        meta_messages.append({
            "role": "user",
            "content": f"{self.meta_prompting_prefix}\n\n{copy.deepcopy(messages[-1]['content'])}\n\n{self.meta_prompting_suffix}"
        })
        
        # Run meta prompter
        meta_outputs, node_id = self.meta_model_generate(
            id=id,
            prompt_or_messages=meta_messages,
            parent_node_ids=parent_node_ids,
            return_node_id=True
        )

        meta_answer = meta_outputs[-1]["content"]

        # Parse answer to specified format
        # VIA MANUAL PARSING
        extracted_answer = meta_answer.split(self.final_answer_indicator)[-1].replace(self.triple_quotes, "").strip()

        if return_node_id:
            return extracted_answer, node_id
        else:
            return extracted_answer

        # VIA LLM CALL
        # Append meta model output to parser messages
        # parser_messages.append({
        #     "role": "assistant",
        #     "content": meta_answer
        # })

        # # append final answer prompt
        # parser_messages.append({
        #     "role": "user",
        #     "content": f"So, what is the final answer? Answer in the specified format."
        # })
        
        # # second call to get final answer
        # return self.completion(
        #     model=self.model_id,
        #     messages=parser_messages,
        #     id=id,
        #     parent_node_ids=node_id,
        #     return_node_id=return_node_id
        # )
    

    def meta_model_generate(
        self,
        id: int,
        prompt_or_messages: Union[str, List[Dict[str, str]]],
        parent_node_ids: Union[int, List[int]] = -1,
        return_node_id: bool = False,
        counter: int = 0
    ) -> Tuple[str, Any]:
        node_id = parent_node_ids
        # This step is defined to ensure that the meta model returns a response in less than 16 rounds.
        # Note: Please feel free to change the number of rounds as you see fit.
        if counter == 16:
            if return_node_id:
                return prompt_or_messages, node_id
            else:
                return prompt_or_messages

        entire_message_log = prompt_or_messages.copy()

        while True:
            entire_message_log[-1][
                "content"
            ] = f"ROUND {counter+1}:\n\n{entire_message_log[-1]['content']}"

            if counter == 14:
                entire_message_log[-1][
                    "content"
                ] += (
                    f"This is the last round; so, please present your final answer."
                )

            # Step 1: Generate an output from the meta model
            meta_model_output, node_id = self.completion(
                model=self.model_id,
                messages=entire_message_log,
                id=id,
                parent_node_ids=parent_node_ids,
                return_node_id=True,
                **self.meta_model_kwargs,
            )

            entire_message_log.append(
                {"role": "assistant", "content": meta_model_output}
            )

            # Check if the meta_model_output contains a text of the form "Expert XYZ:\n" (where XYZ is an alphabanumeric string).

            # Step 2 (a): If we are not in the 0-shot CoT setting, check if the meta model output contains any text between triple quotes.
            # If it does, then generate an output from the corresponding model.
            pattern = r"Expert ((?:\w+ ?){1,5}):\n"
            if (self.fresh_eyes) and (
                # f":\n{self.triple_quotes}" in meta_model_output
                re.search(pattern, meta_model_output)
            ):
                # There might be multiple instructions between the triple quotes; so, split the output by the triple quotes.
                triple_quote_splits = meta_model_output.split(self.triple_quotes)
                # Odd indices are the instructions, even indices contain the lines preceding the instructions (indicating which model to use).
                len_triple_quote_splits = len(triple_quote_splits)

                intermediate_output = ""
                model_num_return_sequences = 1  # Feel free to ignore the model_num_return_sequences > 1 case for now.
                # Iterate over the instructions.
                for i in range(1, len_triple_quote_splits, 2):
                    # Get the instructions for the corresponding model, as well as the line preceding the instructions (indicating which Expert to use).
                    line_preceding_instruction = triple_quote_splits[i - 1].strip()
                    model_name = line_preceding_instruction.split("\n")[-1].strip()
                    if "Expert " in model_name:
                        if model_name[-1] == ":":
                            model_name = model_name[:-1]

                        model_instruction = triple_quote_splits[i].strip()

                        # Add the expert name to the instruction.
                        if self.include_expert_name_in_instruction:
                            model_instruction = (
                                f"You are {model_name}.\n\n{model_instruction}"
                            )

                        # Add "Let's think step by step." to the instruction.
                        if self.use_zero_shot_cot_in_expert_messages:
                            model_instruction += f"\n\nLet's think step by step."

                        # By default, we use the generator Expert to generate an output from the instructions.
                        model_message_list = self.generator_settings["message-list"]

                        current_model_message_list = model_message_list.copy()
                        current_model_message_list.append(
                            {
                                "role": "user",
                                "content": model_instruction,
                            }
                        )

                        if model_name == "Expert Python":
                            current_model_message_list[-1][
                                "content"
                            ] = f"{self.expert_python_message}.\n\n{current_model_message_list[-1]['content']}"

                        # Finally, read to call the corresponding model.
                        model_outputs, node_id = self.completion(
                            model=self.model_id,
                            messages=current_model_message_list,
                            id=id,
                            simple=False,
                            parent_node_ids=node_id,
                            return_node_id=return_node_id,
                            **self.generator_kwargs,
                        ) 

                        for _, model_output in enumerate(model_outputs):
                            ## Special case for Expert Python
                            if model_name == "Expert Python":
                                # If the output contains the special substring, then we need to execute the code.
                                if "Please run this code!" in model_output:
                                    # Get the code #ADDED: 14-08-2023
                                    code_text = model_output.split(
                                        "Please run this code!"
                                    )[0].strip()
                                    # Get the output
                                    code_text = code_text.replace(
                                        "```python", "```"
                                    )
                                    try:
                                        code_text = code_text.split("```")[
                                            -2
                                        ].strip()
                                    except:
                                        code_text = code_text.split("```")[
                                            1
                                        ].strip()

                                    # print(f"We are going to execute the following code:\n{code_text}\n\n")
                                    code_text = rf"{code_text}"
                                    # Execute the code
                                    python_output = safe_execute_subprocess(
                                        code_text,
                                        split_key=None
                                    )
                                    # Add the output to the model output
                                    model_output += f"Here is the Python code used to solve the problem:\n\n{code_text}\n\nHere is the output of the code when executed:\n\n{python_output}"

                            else:
                                specicial_token = "* * *"
                                if self.extract_output:
                                    # FIXME: Temporary fix
                                    if specicial_token in model_output:
                                        model_output = model_output.split(
                                            specicial_token
                                        )[1].strip()

                                    if len(model_output.split(" ")) > 128:
                                        model_output = (
                                            "Solution too long. Please try again."
                                        )
                                else:
                                    model_output.replace(specicial_token, "")
                                    model_output.replace(
                                        "FINAL ANSWER:",
                                        f"{model_name}'s final answer:\n",
                                    )

                            intermediate_output += f"{model_name}'s output:\n{self.triple_quotes}\n{model_output}\n{self.triple_quotes}"

                        # Remove the last two newlines.
                        intermediate_output = intermediate_output.strip()

                        # TODO(msuzgun)[improvement]: Using an additional verifier and/or summarizer might be useful here.
                        # Feel free to ignore the following steps for now (for when model_num_return_sequences > 1).
                        if model_num_return_sequences > 1:
                            summarizer_prompt_or_messages = (
                                self.summarizer_settings["message-list"].copy()
                            )
                            summarizer_prompt_or_messages.append(
                                {
                                    "role": "user",
                                    "content": f"Please provide a clear and concise summary of the expert outputs, emphasizing the key similarities and differences between them.\n\nPrompt: {model_instruction}\n\nOutput: {intermediate_output}",
                                }
                            )

                            # Let's call the summarizer Expert to summarize the outputs.
                            summarizer_output, node_id = self.completion(
                                model=self.model_id,
                                messages=summarizer_prompt_or_messages,
                                id=id,
                                parent_node_ids=node_id,
                                return_node_id=return_node_id,
                                **self.summarizer_kwargs,
                            )

                            # Make this the new intermediate output.
                            intermediate_output = f"Here is the summary of {model_name}'s outputs:\n\n{summarizer_output}"

                # Add the intermediate output to the full prompt or messages.
                intermediate_output = (
                    f"{intermediate_output}\n\n{self.intermediate_feedback}"
                )

                # Add the intermediate output to the full prompt or messages.
                entire_message_log.append(
                    {
                        "role": "user",
                        "content": intermediate_output,
                    }
                )

                # Prepare the prompt for the meta model
                return self.meta_model_generate(
                    id=id,  
                    prompt_or_messages=entire_message_log,
                    parent_node_ids=node_id,
                    return_node_id=return_node_id,
                    counter=counter + 1,
                )
            # Step 2(b): Check if the meta_model_output contains the final answer indicator.
            elif self.final_answer_indicator in meta_model_output:
                # The following code is commented out because we are not using the final answer indicator anymore.
                # However, it is useful for debugging purposes.
                # final_answer = meta_model_output.split(self.final_answer_indicator)[
                #     -1
                # ].strip()
                # print(f"Final answer: {final_answer}")
                if return_node_id:
                    return entire_message_log, node_id
                else:
                    return entire_message_log
            # Step 2(c): We need to continue the (meta-)conversation.
            else:
                entire_message_log.append(
                    {"role": "user", "content": self.error_message}
                )
                return self.meta_model_generate(
                    id=id,
                    prompt_or_messages=entire_message_log,
                    parent_node_ids=node_id,
                    return_node_id=return_node_id,
                    counter=counter + 1
                )
    
    def _set_cfg_from_file(self, cfg_path: str):
        # Load the config
        with open(cfg_path, "r") as f:
            cfg = json.load(f)

        # Set the meta model settings   
        self.meta_model_settings = cfg["meta-model"]
        self.meta_model_kwargs = self.meta_model_settings["parameters"]

        # Set the generator and verifier parameters + summarizer parameters (optional)
        self.generator_settings = cfg["generator"]
        self.generator_kwargs = self.generator_settings["parameters"]
        self.verifier_settings = cfg["verifier"]
        self.verifier_kwargs = self.verifier_settings["parameters"]
        self.summarizer_settings = cfg["summarizer"]
        self.summarizer_kwargs = self.summarizer_settings["parameters"]

        # Set the error message and final answer indicator
        self.error_message = self.meta_model_settings["error-message"]
        self.final_answer_indicator = self.meta_model_settings["final-answer-indicator"]

        # Set the fresh_eyes flag
        self.fresh_eyes = cfg["metadata"]["fresh_eyes"]

        # Other helper variables and constants for the model
        self.triple_quotes = '"""'

        # Set the include_expert_name_in_instruction flag
        self.expert_python_message = cfg["metadata"]["expert_python_message"]
        self.intermediate_feedback = cfg["metadata"]["intermediate_feedback"]
        self.include_expert_name_in_instruction = cfg["metadata"]["include_expert_name_in_instruction"]
        self.extract_output = cfg["metadata"]["extract_output"]
        self.use_zero_shot_cot_in_expert_messages = cfg["metadata"]["use_zero_shot_cot_in_expert_messages"]

        self.meta_prompting_prefix = cfg["metadata"]["meta_prompting_prefix"]
        self.meta_prompting_suffix = cfg["metadata"]["meta_prompting_suffix"]