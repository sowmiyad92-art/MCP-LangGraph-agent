import os
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

def get_tavily_mcp_client():
    """Returns an MCP client configured to connect to Tavily's remote MCP server."""
    client = MultiServerMCPClient(
        {
            "tavily": {
                "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}",
                "transport": "streamable_http",
            }
        }
    )
    return client

async def get_tavily_tools():
    """Fetches the list of tools exposed by Tavily's MCP server."""
    client = get_tavily_mcp_client()
    tools = await client.get_tools()
    return tools