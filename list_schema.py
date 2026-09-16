"""
list_schema.py — see what tables/functions actually exist in Supabase
"""
import os
from supabase import create_client

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase = create_client(url, key)

# List tables via Postgres information_schema (requires an RPC or direct SQL access)
# Easiest path: query pg_proc / information_schema via a raw SQL RPC if you have one,
# otherwise check the Supabase Studio UI: Table Editor + Database > Functions.

# Quick test: try common table names people use with pgvector starters
candidate_tables = ["code_chunks", "documents", "chunks", "embeddings", "kb_chunks"]
for t in candidate_tables:
    try:
        resp = supabase.table(t).select("*").limit(1).execute()
        print(f"[FOUND] table '{t}' exists, sample row: {resp.data}")
    except Exception as e:
        print(f"[MISS]  table '{t}' — {str(e)[:80]}")

candidate_funcs = ["search_code_chunks", "match_documents", "match_chunks"]
for f in candidate_funcs:
    try:
        resp = supabase.rpc(f, {"query_embedding": [0.0]*384, "match_count": 1}).execute()
        print(f"[FOUND] function '{f}' exists, returned {len(resp.data or [])} rows")
    except Exception as e:
        print(f"[MISS]  function '{f}' — {str(e)[:80]}")