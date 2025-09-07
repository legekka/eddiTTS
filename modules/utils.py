import datetime
from dataclasses import dataclass
import json
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from modules.orchestrator import Orchestrator


# OpenAI representation of a tool
@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]

    def as_openai_dict(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

# OpenAI compatible message representation
@dataclass
class Message:
    role: str  # "system", "user", "assistant", "developer", "tool"
    content: Optional[str] = None  # Can be null for tool calls
    name: Optional[str] = None  # For tool calls/responses
    tool_calls: Optional[Dict[str, Any]] = None  # For function calls
    tool_call_id: Optional[str] = None  # For tool responses
    reasoning: Optional[str] = None  # For o1 models reasoning

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format for API calls."""
        message_dict: Dict[str, Any] = {"role": self.role}
        
        if self.content is not None:
            message_dict["content"] = self.content
        if self.name is not None:
            message_dict["name"] = self.name
        if self.tool_calls is not None:
            message_dict["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            message_dict["tool_call_id"] = self.tool_call_id
        if self.reasoning is not None:
            message_dict["reasoning"] = self.reasoning
            
        return message_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create a Message from a dictionary (e.g., from API response)."""
        return cls(
            role=data["role"],
            content=data.get("content"),
            name=data.get("name"),
            tool_calls=data.get("tool_calls"),
            tool_call_id=data.get("tool_call_id"),
            reasoning=data.get("reasoning")
        )

def separate_non_openai_reasoning(content: str) -> Dict[str, str]:
    """Separate non-OpenAI reasoning from the content. Some models use <think>, </think> tokens, which cannot be handled by the OpenAI API.

    Args:
        content (str): The content to process.

    Returns:
        Dict[str, str]: A dictionary with 'reasoning' and 'content' keys.
    """
    reasoning = ""
    content = content.strip()

    # # Check for <think> tags and separate reasoning
    # if "<think>" in content and "</think>" in content:
    #     reasoning = content.split("<think>", 1)[1].split("</think>", 1)[0].strip()
    #     content = content.replace(f"<think>{reasoning}</think>", "").strip()

    if "</think>" in content:
        reasoning = content.split("</think>", 1)[0]
        reasoning = reasoning.replace("<think>", "").strip()
        content = content.split("</think>", 1)[1].strip()
    else:
        reasoning = ""

    return {"reasoning": reasoning, "content": content}