"""Module for interfacing with OpenAI-compatible LLM backends.

This module defines the LlmBackend class for communication with OpenAI-compatible APIs,
managing available models and chat completions with optional tool support.
"""

import openai
from modules.utils import Tool, separate_non_openai_reasoning
from typing import Any, Dict, List, Optional


class LlmBackend:
    """Manages connection to an OpenAI-compatible LLM backend.

    Initializes the API client, retrieves available models, and handles chat completions
    with optional function (tool) support.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        api_key: Optional[str] = None,
    ):
        """Initialize the LlmBackend.

        Args:
            base_url (str, optional): Base URL for the OpenAI-compatible API. Defaults to the local endpoint.
            api_key (str, optional): API key for authentication. Defaults to None.

        Raises:
            RuntimeError: If the API client fails to initialize.
        """
        self.base_url = base_url
        self.api_key = api_key

        try:
            # Use dummy API key if none provided (for local endpoints that don't require auth)
            api_key = self.api_key if self.api_key else "dummy-key"
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url=self.base_url,
                timeout=3600,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI-compatible API client: {e}")

        self.models = self.get_models()

    def get_models(self):
        """Fetch and return the list of available models from the backend.

        Returns:
            List[Any]: List of model objects returned by the API client.

        Raises:
            RuntimeError: If the model list retrieval fails.
        """
        try:
            return self.client.models.list().data
        except Exception as e:
            raise RuntimeError(f"Failed to update models: {e}")
        
    def list_models(self) -> List[str]:
        """Return IDs of all available models.

        Returns:
            List[str]: List of model ID strings.
        """
        return [model.id for model in self.models]

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        reasoning_effort: Optional[str],
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[str] | Dict[str, str] = "auto",
        **kwargs,
    ):
        """Send a chat completion request to the LLM backend.

        Args:
            messages (List[Dict[str, str]]): Sequence of message dicts with 'role' and 'content'.
            model (str): Model ID to use for the chat completion.
            tools (Optional[List[Tool]]): Tools available for function calls. Defaults to None.
            tool_choice (Union[str, Dict[str, str]], optional): Strategy or explicit tool selection. Defaults to 'auto'.
            reasoning_effort (str): Reasoning effort of the communication
            **kwargs: Additional parameters passed to the API client.

        Returns:
            Any: The response object from the chat API, including choices and function calls.

        Raises:
            RuntimeError: If the chat completion request fails.
        """
        params = {
            "model": model,
            "reasoning_effort": reasoning_effort,
            "messages": messages,
            "tools": [t.as_openai_dict() for t in tools] if tools else None,
            "tool_choice": tool_choice,
            **kwargs,
        }

        try:
            response = self.client.chat.completions.create(**params)
        except Exception as e:
            raise RuntimeError(f"Failed to create chat completion: {e}")
        
        if (response.choices[0].message.content and 
            "</think>" in response.choices[0].message.content):
            message = separate_non_openai_reasoning(response.choices[0].message.content)
            response.choices[0].message.content = message["content"]
            response.choices[0].message.reasoning = message["reasoning"]
        elif hasattr(response.choices[0].message, 'reasoning_content') and response.choices[0].message.reasoning_content:
            response.choices[0].message.reasoning = response.choices[0].message.reasoning_content
        if (not response.choices[0].message.content):
            response.choices[0].message.content = ""

        return response