# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict

import openai
import requests
import tiktoken
from openai.types.chat import ChatCompletion, ChatCompletionMessage

from camel.typing import ModelType
from chatdev.statistics import prompt_cost, DEFAULT_AI_MODEL
from chatdev.utils import log_visualize

# try:
#     from openai.types.chat import ChatCompletion, ChatCompletionMessage
#
#     openai_new_api = True  # new openai api version
# except ImportError:
#     openai_new_api = False  # old openai api version

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
if 'BASE_URL' in os.environ:
    BASE_URL = os.environ['BASE_URL']
else:
    BASE_URL = None

DEFAULT_AI_MODEL = os.environ.get('DEFAULT_AI_MODEL')
if not DEFAULT_AI_MODEL:
    DEFAULT_AI_MODEL = "deepseek-r1-250120"

AI_PROVIDER = os.getenv("AI_PROVIDER", "openai")  # Default to OpenAI if not set


class ModelBackend(ABC):
    r"""Base class for different model backends.
    May be OpenAI API, a local LLM, a stub for unit tests, etc."""

    @abstractmethod
    def run(self, *args, **kwargs):
        r"""Runs the query to the backend model.

        Raises:
            RuntimeError: if the return value from OpenAI API
            is not a dict that is expected.

        Returns:
            Dict[str, Any]: All backends must return a dict in OpenAI format.
        """
        pass


class OpenAIModel(ModelBackend):
    r"""OpenAI API in a unified ModelBackend interface."""

    def __init__(self, model_type: ModelType, model_config_dict: Dict) -> None:
        super().__init__()
        self.model_type = model_type
        self.model_config_dict = model_config_dict

    def run(self, *args, **kwargs):
        string = "\n".join([message["content"] for message in kwargs["messages"]])
        # encoding = tiktoken.encoding_for_model(self.model_type.value)
        encoding = tiktoken.get_encoding("cl100k_base")  # Use an encoding compatible with OpenAI models
        num_prompt_tokens = len(encoding.encode(string))
        gap_between_send_receive = 15 * len(kwargs["messages"])
        num_prompt_tokens += gap_between_send_receive

        client = openai.OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=BASE_URL,
        )

        num_max_token_map = {
            # "gpt-3.5-turbo": 4096,
            # "gpt-3.5-turbo-16k": 16384,
            # "gpt-3.5-turbo-0613": 4096,
            # "gpt-3.5-turbo-16k-0613": 16384,
            # "gpt-4": 8192,
            # "gpt-4-0613": 8192,
            # "gpt-4-32k": 32768,
            # "gpt-4-turbo": 100000,
            # "gpt-4o": 4096,  # 100000
            # "gpt-4o-mini": 16384,  # 100000
            "deepseek-r1-250120": 16384,
        }
        num_max_token = num_max_token_map[
            self.model_type.value] if self.model_type.value in num_max_token_map else 4096
        num_max_completion_tokens = num_max_token - num_prompt_tokens
        self.model_config_dict['max_tokens'] = num_max_completion_tokens

        if AI_PROVIDER == "openai":
            response = client.chat.completions.create(*args, **kwargs, model=self.model_type.value,
                                                      **self.model_config_dict)
        else:
            # print(call_ai(model=DEFAULT_AI_MODEL, messages=[{"role": "user", "content": "Hello!"}]))
            response = call_ollama(*args, **kwargs)

        num_prompt_tokens = response.usage.prompt_tokens
        num_completion_tokens = response.usage.completion_tokens

        cost = prompt_cost(
            self.model_type.value,
            num_prompt_tokens=num_prompt_tokens,
            # num_prompt_tokens=response.get("usage", {}).get("prompt_tokens", 1),
            num_completion_tokens=num_completion_tokens
            # num_completion_tokens=response.get("usage", {}).get("completion_tokens", 2)
        )

        log_visualize(
            "**[{} Usage_Info Receive]**\nprompt_tokens: {}\ncompletion_tokens: {}\ntotal_tokens: {}\ncost: ${:.6f}\n".format(
                BASE_URL,
                num_prompt_tokens,
                num_completion_tokens,
                num_prompt_tokens + num_completion_tokens, cost))
        if not isinstance(response, ChatCompletion):
            raise RuntimeError("Unexpected return from OpenAI API")
        return response


class StubModel(ModelBackend):
    r"""A dummy model used for unit tests."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def run(self, *args, **kwargs) -> Dict[str, Any]:
        ARBITRARY_STRING = "Lorem Ipsum"

        return dict(
            id="stub_model_id",
            usage=dict(),
            choices=[
                dict(finish_reason="stop",
                     message=dict(content=ARBITRARY_STRING, role="assistant"))
            ],
        )


class ModelFactory:
    r"""Factory of backend models.

    Raises:
        ValueError: in case the provided model type is unknown.
    """

    @staticmethod
    def create(model_type: ModelType, model_config_dict: Dict) -> ModelBackend:
        default_model_type = ModelType.DEFAULT_MODEL

        if model_type in {
            # ModelType.GPT_3_5_TURBO,
            # ModelType.GPT_3_5_TURBO_NEW,
            # ModelType.GPT_4,
            # ModelType.GPT_4_32k,
            # ModelType.GPT_4_TURBO,
            # ModelType.GPT_4_TURBO_V,
            # ModelType.GPT_4O,
            # ModelType.GPT_4O_MINI,
            ModelType.DEFAULT_MODEL,
            None
        }:
            model_class = OpenAIModel
        elif model_type == ModelType.STUB:
            model_class = StubModel
        else:
            raise ValueError("Unknown model")

        if model_type is None:
            model_type = default_model_type

        # log_visualize("Model Type: {}".format(model_type))
        inst = model_class(model_type, model_config_dict)
        return inst


from pydantic import BaseModel  # Ensure compatibility with Pydantic


class OllamaChatCompletion(ChatCompletion, BaseModel):
    """A wrapper to make Ollama responses behave like OpenAI's ChatCompletion."""

    id: str
    object: str
    created: int
    model: str
    usage: dict
    choices: list

    def __init__(self, response, model_name):
        super().__init__(  # ✅ Properly initialize the Pydantic model
            id="ollama-123",
            object="chat.completion",
            created=int(time.time()),
            model=model_name,
            usage={
                "prompt_tokens": 1,  # Ollama does not provide token usage
                "completion_tokens": 2,
                "total_tokens": 3
            },
            choices=[
                {
                    "message": ChatCompletionMessage(  # ✅ Convert to an OpenAI-style object
                        role="assistant",
                        content=response.get("response", "No response")
                    ),
                    "finish_reason": "stop",
                    "index": 0
                }
            ]
        )

    # ✅ Override __getitem__ to mimic a dictionary
    def __getitem__(self, key):
        return getattr(self, key, None)

    def get(self, key, default=None):
        """Mimic dictionary get() behavior"""
        return getattr(self, key, default)


def call_ollama(*args, **kwargs):
    model_name = kwargs.get("model", DEFAULT_AI_MODEL)
    prompt = kwargs.get("messages", [{"role": "user", "content": "Hello!"}])[-1]["content"]

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(BASE_URL, json=payload)

    try:
        ollama_response = response.json()
        return OllamaChatCompletion(ollama_response, model_name)  # ✅ Return a structured OpenAI-style object
    except Exception as e:
        print("JSON Decode Error:", e)
        return OllamaChatCompletion({"response": "Error processing request"}, model_name)
