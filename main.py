"""
Main FastAPI server combining LangGraph router (sql|pdf|web|mcp) + Knowledge Base search
Endpoints:
  - GET /health — health check
  - POST /query — LangGraph router (sql|pdf|web|mcp)
  - POST /api/kb-search — Knowledge Base search
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from graph import graph  # Your LangGraph StateGraph
from knowledge_base_node import create_kb_router

load_dotenv()

# ============================================================================
# FASTAPI APP SETUP
# ============================================================================

app = FastAPI(
    title="LangGraph Router + Knowledge Base API",
    description="Multi-route agent (sql|pdf|web|mcp) + code knowledge base search",
    version="1.0"
)

# CORS middleware (allow Aadsia frontend to call)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class QueryRequest(BaseModel):
    """Request body for /query endpoint (LangGraph router)."""
    query: str = Field(..., min_length=1, max_length=1000, description="User query")

class QueryResponse(BaseModel):
    """Response body for /query endpoint."""
    query: str
    route: str  # "sql" | "pdf" | "web" | "mcp"
    answer: str
    source_used: str
    metadata: dict | None = None

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    message: str

# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "LangGraph Router + KB API is running"
    }

# ============================================================================
# LANGGRAPH QUERY ENDPOINT
# ============================================================================

@app.post("/query", response_model=QueryResponse)
async def query_langgraph(request: QueryRequest):
    """
    Route query through LangGraph router.
    Automatically classifies as: sql | pdf | web | mcp
    
    Args:
        query: User's question
    
    Returns:
        answer: Result from the routed node
        route: Which route was triggered
        source_used: Data source (imdb, pdf, web, mcp)
    """
    try:
        # Invoke LangGraph with query
        state_input = {"messages": [], "query": request.query}
        
        # Execute graph (synchronous)
        result = graph.invoke(state_input)
        
        # Extract route info
        route = result.get("route", "unknown")
        answer = result.get("result", "No answer returned")
        source = result.get("source_used", "unknown")
        
        return QueryResponse(
            query=request.query,
            route=route,
            answer=answer,
            source_used=source,
            metadata={
                "route_confidence": result.get("route_confidence"),
                "processing_time_ms": result.get("processing_time")
            }
        )
    
    except Exception as e:
        print(f"[QUERY] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# INCLUDE KB ROUTER
# ============================================================================

kb_router = create_kb_router()
app.include_router(kb_router)

# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/")
async def root():
    """API root — available endpoints."""
    return {
        "status": "running",
        "endpoints": {
            "health": "GET /health",
            "langgraph_query": "POST /query (routes to sql|pdf|web|mcp)",
            "kb_search": "POST /api/kb-search (code knowledge base)",
            "docs": "GET /docs (interactive Swagger UI)"
        }
    }

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=True
    )