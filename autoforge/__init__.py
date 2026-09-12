"""autoforge — an agent framework whose tools are data, and whose self-made
tools must earn the right to be called.

Design pillars
--------------
1. Tools are data, not code locks (hot-loadable, no restart).
2. A tool is born with a *contract*: schema + trigger probes + effect signature.
3. The forge pipeline verifies a tool before activation and re-verifies it
   while it lives (reliability ledger, auto-quarantine).
4. Everything is pluggable: llm, sandbox, ledger, router.

Freedom is the default; verification is a capability you can attach, not a
cage the agent is born inside.
"""

__version__ = "0.1.0"

from autoforge.core.llm import LLMClient, MockLLMClient, OpenAICompatClient
from autoforge.core.message import Message, ToolCall
from autoforge.tools.registry import ToolRegistry, ToolResult
from autoforge.tools.spec import ToolSpec, ToolState, ToolStats, TriggerProbe

__all__ = [
    "__version__",
    "LLMClient",
    "OpenAICompatClient",
    "MockLLMClient",
    "Message",
    "ToolCall",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "ToolState",
    "ToolStats",
    "TriggerProbe",
]
