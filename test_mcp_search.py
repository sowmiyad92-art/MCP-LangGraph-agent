import asyncio
from agents.mcp_search_node import run_mcp_search

async def main():
    query = "What is the latest quarterly earnings date announced for Netflix?"
    answer = await run_mcp_search(query)
    print("ANSWER:\n", answer)

if __name__ == "__main__":
    asyncio.run(main())