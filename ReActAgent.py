from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import os

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a: int, b: int) -> int:
    """Add two numbers together.
    
    Args:
        a: First number to add
        b: Second number to add
        
    Returns:
        The sum of a and b
    """
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtracts two numbers together.
    
    Args:
        a: First number to add
        b: Second number to add
        
    Returns:
        The subtraction of a and b
    """
    return a - b


tools = [add,subtract]

model = ChatOpenAI(
    model="deepseek/deepseek-chat-v3.1:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),  
    openai_api_key=os.getenv("OPENROUTER_API_KEY")
).bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    systemPrompt = SystemMessage(content="You are 'JOD' a helpful assistant.")
    response = model.invoke([systemPrompt] + state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    }
)

graph.add_edge("tools", "our_agent")
app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [HumanMessage(content="What is 2+2 also subtract 3 from it then write a small joke on maths.")]}
print_stream(app.stream(inputs, stream_mode="values"))