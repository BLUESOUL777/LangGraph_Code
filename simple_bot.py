from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os
import requests
from types import SimpleNamespace

load_dotenv()

class AgentState(TypedDict):
    messages: List[HumanMessage]  # Fixed: changed = to :

class OpenRouterLLM:
    def __init__(self, model: str, temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self.api_key = os.getenv("OPENROUTER_API_KEY")

    def __call__(self, messages: List[HumanMessage]):
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY not found in environment")
        payload_messages = [{"role": "user", "content": m.content} for m in messages]
        payload = {
            "model": self.model,
            "messages": payload_messages,
            "temperature": self.temperature
        }
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=60
        )
        resp.raise_for_status()
        data = resp.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except Exception:
            content = str(data)
        return SimpleNamespace(content=content)

llm = OpenRouterLLM(model="deepseek/deepseek-chat-v3.1:free", temperature=0)

def process(state: AgentState) -> AgentState:
    response = llm(state["messages"])
    print(f"\nAI :- {response.content}\n")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

user_input = input("Enter --> ")
agent.invoke({"messages": [HumanMessage(content=user_input)]})