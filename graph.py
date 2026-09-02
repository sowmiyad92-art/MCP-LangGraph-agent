import os
import asyncio
import pandas as pd
from typing import TypedDict, Optional
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from langchain_core.tools import StructuredTool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END
from tavily import TavilyClient
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_huggingface import HuggingFaceEmbeddings
from supabase import create_client
from groq import Groq

from agents.mcp_search_node import run_mcp_search

load_dotenv()

# ================= State =================

class RouterState(TypedDict):
    messages: list
    query: str
    route: str          # "sql" | "pdf" | "web" | "mcp"
    result: str
    source_used: str


# ================= Shared clients =================

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)


# ================= Classifier node =================

CLASSIFIER_SYSTEM_PROMPT = """You are a routing classifier. Given a user query, respond with
exactly one word: sql, pdf, web, or mcp.

- sql: query is about structured movie/IMDb data (ratings, votes, release year, genre,
  counts, lists, lookups) answerable from a database. This includes questions like
  "how many movies came out in [year]" or "list movies from [genre]".
- pdf: query asks about content inside an uploaded document (summarize, extract, explain
  a section, risk factors, etc.).
- web: query needs current/live, real-world information using a single quick search
  (breaking news, today's events, a specific fact, a quick lookup).
- mcp: query explicitly asks for deep/detailed/comprehensive research, multi-source
  synthesis, or extracting/crawling content from a specific URL. Trigger words: "detailed",
  "in-depth", "comprehensive", "research summary", "analyze in depth", a pasted URL.

Examples:
"How many movies were released in 2020?" -> sql
"List top rated sci-fi movies from 2015" -> sql
"Summarize the risk factors section" -> pdf
"What does this document say about revenue?" -> pdf
"What's the latest news on Netflix's Q2 earnings?" -> web
"Who won the Oscar for best picture last night?" -> web
"Give me a detailed research summary on Netflix's content strategy" -> mcp
"Extract the content from https://example.com/article" -> mcp
"Give me a detailed research summary on Netflix's content strategy" -> mcp
"What's the latest news on Netflix's Q2 earnings?" -> web

Respond with only the single word. No punctuation, no explanation.
"""

def classifier_node(state: RouterState) -> RouterState:
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": state["query"]},
        ],
        max_tokens=300,
        temperature=0,
    )

    raw = response.choices[0].message.content
    print(f"[CLASSIFIER RAW OUTPUT]: {raw!r}")   # temporary debug line

    route = (raw or "").strip().lower()
    if route not in ("sql", "pdf", "web", "mcp"):
        route = "web"
    state["route"] = route
    return state

def route_decision(state: RouterState) -> str:
    return state["route"]


# ================= SQL node (reuses genai-search-agent tools) =================

# ---- IMDb lookup tool ----
class ImdbQuery(BaseModel):
    genre: Optional[str] = Field(None, description="Genre to filter by, e.g. Horror, Drama")
    min_votes: Optional[int] = Field(None, description="Minimum number of votes")
    year_from: Optional[int] = Field(None, description="Earliest release year")
    year_to: Optional[int] = Field(None, description="Latest release year")

DF = pd.read_csv("imdb_filtered_2000_2024.csv")
DF["startYear"] = DF["startYear"].fillna(0).astype(int)
DF["numVotes"] = DF["numVotes"].fillna(0).astype(int)

def imdb_lookup(genre: Optional[str] = None, min_votes: Optional[int] = None,
                 year_from: Optional[int] = None, year_to: Optional[int] = None) -> str:
    df = DF.copy()
    if genre:
        df = df[df["genres"].str.contains(genre, case=False, na=False)]
    if min_votes:
        df = df[df["numVotes"] >= min_votes]
    if year_from:
        df = df[df["startYear"] >= year_from]
    if year_to:
        df = df[df["startYear"] <= year_to]
    if df.empty:
        return "No movies found matching those filters."
    df = df.sort_values("averageRating", ascending=False).head(10)
    lines = [
        f"{r['primaryTitle']} ({r['startYear']}) - Rating {r['averageRating']}, Votes {r['numVotes']}"
        for _, r in df.iterrows()
    ]
    return "\n".join(lines)

imdb_tool = StructuredTool.from_function(
    func=imdb_lookup,
    name="imdb_lookup",
    description="Query a dataset of 30,000+ movies (2000-2024) by genre, vote count, or year range.",
    args_schema=ImdbQuery,
)

# ---- SQL toolkit ----
db = SQLDatabase.from_uri("sqlite:///imdb.db")
sql_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
sql_tools = SQLDatabaseToolkit(db=db, llm=sql_llm).get_tools()

SQL_SYSTEM_INSTRUCTION = "You are a movie research assistant with access to an IMDb dataset and SQL database."

def get_sql_agent():
    return create_react_agent(model=llm, tools=[imdb_tool] + sql_tools, prompt=SQL_SYSTEM_INSTRUCTION)

def sql_node(state: RouterState) -> RouterState:
    agent_executor = get_sql_agent()
    final_content = ""
    for event in agent_executor.stream(
        {"messages": [("human", state["query"])]},
        stream_mode="values",
        config={"recursion_limit": 50},
    ):
        if "messages" in event:
            last_msg = event["messages"][-1]
            if last_msg.type == "ai" and not getattr(last_msg, "tool_calls", None):
                final_content = last_msg.content

    state["result"] = final_content
    state["source_used"] = "sql"
    return state


# ================= PDF node (reuses pdf-rag-agent rag_answer logic) =================

def pdf_node(state: RouterState) -> RouterState:
    question = state["query"]
    query_vector = embeddings.embed_query(question)
    result = supabase.rpc("match_documents", {
        "query_embedding": query_vector,
        "match_count": 3,
    }).execute()

    context = "\n\n".join(row["content"] for row in result.data)
    prompt = f"""Answer the question using only the context below. If the answer isn't in the context, say so.

Context:
{context}

Question: {question}
"""
    response = llm.invoke(prompt)

    state["result"] = response.content
    state["source_used"] = "pdf"
    return state


# ================= Web node (reuses Tavily from genai-search-agent) =================

def web_research(query: str) -> str:
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        return "Tavily API key not configured."
    try:
        client = TavilyClient(api_key=key)
        results = client.search(query, max_results=4)
        chunks = []
        for r in results.get("results", []):
            chunks.append(f"{r.get('title','Untitled')}: {r.get('content','')[:300]}")
        return "\n\n".join(chunks) if chunks else "No results found."
    except Exception as e:
        return f"Search failed: {e}"

def web_node(state: RouterState) -> RouterState:
    state["result"] = web_research(state["query"])
    state["source_used"] = "web"
    return state


# ================= MCP node (deep research / URL extraction) =================

def mcp_node(state: RouterState) -> RouterState:
    state["result"] = asyncio.run(run_mcp_search(state["query"]))
    state["source_used"] = "mcp"
    return state


# ================= Graph assembly =================

def build_graph():
    graph = StateGraph(RouterState)
    graph.add_node("classifier", classifier_node)
    graph.add_node("sql", sql_node)
    graph.add_node("pdf", pdf_node)
    graph.add_node("web", web_node)
    graph.add_node("mcp", mcp_node)

    graph.add_edge(START, "classifier")
    graph.add_conditional_edges("classifier", route_decision, {
        "sql": "sql",
        "pdf": "pdf",
        "web": "web",
        "mcp": "mcp",
    })
    graph.add_edge("sql", END)
    graph.add_edge("pdf", END)
    graph.add_edge("web", END)
    graph.add_edge("mcp", END)

    return graph.compile()