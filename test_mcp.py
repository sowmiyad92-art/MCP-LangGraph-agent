import asyncio
from mcp_servers.tavily_client import get_tavily_tools

async def main():
    all_tools = await get_tavily_tools()
    tools = [t for t in all_tools if t.name in ("tavily_search", "tavily_extract")])
    print(f"Loaded {len(tools)} tools from Tavily MCP:")
    for t in tools:
        print(f"- {t.name}: {t.description}")

if __name__ == "__main__":
    asyncio.run(main())