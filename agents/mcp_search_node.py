import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from mcp_servers.tavily_client import get_tavily_tools

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

async def run_mcp_search(query: str) -> str:
    tools = await get_tavily_tools()
    search_tool = next(t for t in tools if t.name == "tavily_search")

    raw_result = await search_tool.ainvoke({"query": query})
    trimmed = str(raw_result)[:3000]

    prompt = f"Based on the search results below, answer the question concisely.\n\nSearch results:\n{trimmed}\n\nQuestion: {query}"
    response = await llm.ainvoke(prompt)
    return response.content
