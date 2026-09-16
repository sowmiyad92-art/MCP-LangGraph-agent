"""
diagnose_kb.py — standalone diagnostic for the knowledge_base_node search path.
Run: python diagnose_kb.py
"""

import os
import traceback
from sentence_transformers import SentenceTransformer
from supabase import create_client

# ---- 1. Check env vars ----
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
print(f"[1] SUPABASE_URL set: {bool(url)}")
print(f"[1] SUPABASE_KEY set: {bool(key)}")
if not url or not key:
    raise SystemExit("Missing env vars — check your .env / export")

# ---- 2. Connect to Supabase ----
try:
    supabase = create_client(url, key)
    print("[2] Supabase client created OK")
except Exception:
    print("[2] FAILED to create Supabase client:")
    traceback.print_exc()
    raise SystemExit()

# ---- 3. Check table row count directly ----
try:
    # replace 'code_chunks' with your actual table name if different
    resp = supabase.table("code_chunks").select("id", count="exact").limit(1).execute()
    print(f"[3] Row count in code_chunks: {resp.count}")
except Exception:
    print("[3] FAILED to query table directly (check table name):")
    traceback.print_exc()

# ---- 4. Check embedding dimension of one stored row ----
try:
    resp = supabase.table("code_chunks").select("id, embedding").limit(1).execute()
    if resp.data:
        emb = resp.data[0].get("embedding")
        if emb:
            # embedding may come back as a string like "[0.1,0.2,...]" or a list
            if isinstance(emb, str):
                dim = emb.count(",") + 1
            else:
                dim = len(emb)
            print(f"[4] Stored embedding dimension: {dim}")
        else:
            print("[4] Row found but embedding column is NULL")
    else:
        print("[4] No rows returned — table may be empty")
except Exception:
    print("[4] FAILED to inspect embedding column:")
    traceback.print_exc()

# ---- 5. Generate a query embedding and check its dimension ----
print("[5] Loading sentence-transformers model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
query = "how does the classifier route queries"
query_embedding = model.encode(query, convert_to_tensor=False).tolist()
print(f"[5] Query embedding dimension: {len(query_embedding)}")

# ---- 6. Call the RPC directly and print the FULL error if it fails ----
print("[6] Calling search_code_chunks RPC...")
try:
    response = supabase.rpc(
        "search_code_chunks",
        {
            "query_embedding": query_embedding,
            "match_count": 3
        }
    ).execute()
    print(f"[6] RPC succeeded. Rows returned: {len(response.data) if response.data else 0}")
    if response.data:
        for row in response.data:
            print("   ", row)
    else:
        print("[6] RPC returned zero rows — check match_threshold inside the SQL function")
except Exception:
    print("[6] RPC FAILED — full traceback:")
    traceback.print_exc()

print("\nDone.")