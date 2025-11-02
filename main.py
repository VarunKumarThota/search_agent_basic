from dotenv import load_dotenv
import httpx

from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

# Import from LangGraph style (not the old AgentExecutor / react agent import path).
from langchain.agents import create_agent # newer API
# Note: depending on version, the import path might be:
#   from langgraph.prebuilt import create_react_agent
# or
#   from langchain.graphs.agents import create_react_agent
# Check your installed version’s docs.

load_dotenv()

# Setup tool(s)
tools = [TavilySearch()]

# Setup LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1,
    http_client=httpx.Client(verify=False)
)

# Create agent using the new API
agent = create_agent(
    model=llm,
    tools=tools,
    # Optionally provide a prompt or system message or custom state initializer
    system_prompt="You are an assistant that uses these tools: {tools}. When you need data, call a tool, etc."
)

def main():
    print("Hello from search-agent!")
    # Note with LangGraph you usually pass messages rather than a dict with “input”
    result = agent.invoke(
        "Search 3 job postings for an AI engineer in India using LinkedIn and list their details."
    )

    print(result)

if __name__ == "__main__":
    main()
