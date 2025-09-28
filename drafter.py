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

document_content = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def update(content: str) -> str:
    """Update the global document_content variable with new content."""
    global document_content
    document_content = content
    return f"The current content of the document is --> {document_content}"

@tool 
def save(filename: str) -> str:
    """Save the document to a text file and finish the process.
    
    Args:
        filename: Name of the text file.
    """
    global document_content

    if not filename.endswith(".txt"):
        filename += ".txt"

    try:
        with open(filename, "w") as file:
            file.write(document_content)
        return f"Document saved successfully as {filename}."
    except Exception as e:
        return f"Failed to save document: {str(e)}"

tools = [update, save]

model = ChatOpenAI(
    model="deepseek/deepseek-chat-v3.1:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),  
    openai_api_key=os.getenv("OPENROUTER_API_KEY")
).bind_tools(tools)

def our_agent(state: AgentState) -> AgentState:
    systemPrompt = SystemMessage(content=f"You are 'DrafterGPT' a helpful assistant that helps users draft documents. You have two tools at your disposal: 'update' and 'save'. Use 'update' to add or modify content in the document. Use 'save' to save the document to a text file and finish the process. Always use 'save' when the user indicates they are done drafting. The file content is {document_content}.")
    
    if not state["messages"]:
        user_input = "I want to draft a document. So help me get started."
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("What would you like to do with the document? ")
        print(f"USER :- {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [systemPrompt] + list(state["messages"]) + [user_message]

    response = model.invoke(all_messages)

    print(f"DrafterGPT :- {response.content}")
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"Tool Calls :- {', '.join([tc['name'] for tc in response.tool_calls])}")
    
    return {"messages": list(state["messages"]) + [user_message, response]}

def should_continue(state: AgentState) -> str:
    """Determine whether to continue or end the process based on the last message."""
    if not state["messages"]:
        return "continue"
    
    # FIXED: Use 'name' instead of 'tool_name'
    for message in reversed(state["messages"]):
        if isinstance(message, ToolMessage) and message.name == "save":
            return "end"
        
    return "continue"

def print_message(messages: Sequence[BaseMessage]):
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"Tool Content :- {message.content}")

graph = StateGraph(AgentState)

graph.add_node("our_agent", our_agent)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("our_agent")
graph.add_edge("tools", "our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    }
)

app = graph.compile()

def run_document_drafter():
    print("\n ======Drafter GPT====== \n")
    state: AgentState = {"messages": []}
    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_message(step["messages"])

    print("\n ======End of Session====== \n")

if __name__ == "__main__":
    run_document_drafter()