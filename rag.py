from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, BaseMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool

# Set environment variable to avoid tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

llm = ChatOpenAI(
    model="deepseek/deepseek-chat-v3.1:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# Use updated HuggingFace embeddings
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

pdf_path = "stock_market_report.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path)

try:
    pages = pdf_loader.load()
    print(f"PDF has been loaded successfully with {len(pages)} pages.")
except Exception as e:
    print(f"Error loading PDF: {e}")
    raise

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
pages_split = text_splitter.split_documents(pages)

persist_directory = "chroma_db"
collection_name = "stock_market"

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

try:
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings_model,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print(f"Created Chroma db vector store!")
except Exception as e:
    print(f"Error creating Chroma db vector store: {e}")
    raise

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

@tool
def retriever_tool(query: str) -> str:
    """This tool retrieves relevant documents based on the query."""
    try:
        docs = retriever.invoke(query)

        if not docs:
            return "I found no relevant information in the document provided."
        
        results = []
        for i, doc in enumerate(docs):
            results.append(f"Document {i+1}:\n{doc.page_content}\n")

        return "\n\n".join(results)
    except Exception as e:
        return f"Error retrieving documents: {str(e)}"

tools = [retriever_tool]

llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState) -> str:
    """Determine whether the last message has any tool call."""    
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

system_prompt = """You are an intelligent AI assistant that answers questions based on the context provided from a PDF document.
You have access to a tool called 'retriever_tool' that you can use to retrieve relevant information from the document.
Use the tool when you need to find specific information from the document to answer the user's question.
If the user asks a question that is not related to the document, politely inform them that you can only answer questions related to the document.
Always use the tool when you need to find specific information from the document.
"""

tools_dict = {tool.name: tool for tool in tools}

def call_llm(state: AgentState) -> AgentState:
    """Function to call the llm with the current state messages."""
    messages = list(state["messages"])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)
    return {'messages': [message]}

def take_action(state: AgentState) -> AgentState:
    """Execute the tool calls from the llm's response""" 
    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")

        if t['name'] not in tools_dict:
            print(f"Tool doesn't exist.")
            result = "Incorrect tool name please select a valid tool from the list of the provided tools."
        else:
            result = tools_dict[t['name']].invoke(t['args'].get("query", ''))
            print(f"Result length: {len(str(result))}")

        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Tools Execution Complete. Back to the model.")
    return {'messages': results}

graph = StateGraph(AgentState)
graph.add_node("call_llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "call_llm",
    should_continue,
    {
        True: "retriever_agent",
        False: END,
    }
)
graph.add_edge("retriever_agent", "call_llm")
graph.set_entry_point("call_llm")

rag_agent = graph.compile()

def running_agent():
    print("\n" + "="*6 + " RAG GPT " + "="*6 + "\n")
    
    while True:
        user_input = input("What is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        if not user_input.strip():
            print("Please enter a valid question.")
            continue

        messages = [HumanMessage(content=user_input)]
        result = rag_agent.invoke({"messages": messages})
        print("\n" + "="*5 + " RAG GPT ANSWER " + "="*5 + "\n")
        print(result['messages'][-1].content)
        print("\n" + "="*40 + "\n")

if __name__ == "__main__":
    running_agent()