"""
ingest_code.py — chunk local repo files, embed them, and load into Supabase code_chunks
Run: python ingest_code.py
"""

import os
import glob
from sentence_transformers import SentenceTransformer
from supabase import create_client

# ---- CONFIG ----
REPO_PATHS = {
    "mcp-langgraph-agent": r"C:\Users\sowmiya\OneDrive\Documents\mcp-langgraph-agent",
    # add more repos here: "other-repo": r"C:\path\to\other-repo",
}
FILE_EXTENSIONS = (".py", ".js", ".ts", ".md")  # extend as needed
EXCLUDE_DIRS = {"venv", "node_modules", ".git", "__pycache__", "dist", "build"}
CHUNK_LINES = 60       # lines per chunk
CHUNK_OVERLAP = 10     # overlapping lines between chunks
BATCH_SIZE = 50        # rows per Supabase insert

# ---- INIT ----
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase = create_client(url, key)

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")


def chunk_file(filepath: str, chunk_lines=CHUNK_LINES, overlap=CHUNK_OVERLAP):
    """Split a file into overlapping line-based chunks."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"  [skip] could not read {filepath}: {e}")
        return []

    chunks = []
    i = 0
    n = len(lines)
    if n == 0:
        return []

    while i < n:
        end = min(i + chunk_lines, n)
        snippet = "".join(lines[i:end]).strip()
        if snippet:
            chunks.append({
                "content": snippet,
                "line_range": f"{i+1}-{end}"
            })
        if end == n:
            break
        i += (chunk_lines - overlap)

    return chunks


def collect_files(root: str):
    """Walk directory, skip excluded dirs, return matching file paths."""
    matched = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fname in filenames:
            if fname.endswith(FILE_EXTENSIONS):
                matched.append(os.path.join(dirpath, fname))
    return matched


def main():
    all_rows = []

    for project_name, repo_path in REPO_PATHS.items():
        print(f"\nScanning {project_name} at {repo_path}...")
        files = collect_files(repo_path)
        print(f"  Found {len(files)} files")

        for filepath in files:
            rel_path = os.path.relpath(filepath, repo_path).replace("\\", "/")
            chunks = chunk_file(filepath)
            for c in chunks:
                all_rows.append({
                    "content": c["content"],
                    "project_name": project_name,
                    "file_path": rel_path,
                    "line_range": c["line_range"],
                })

    print(f"\nTotal chunks to embed: {len(all_rows)}")
    if not all_rows:
        print("Nothing to ingest — check REPO_PATHS / FILE_EXTENSIONS.")
        return

    # Embed in batches (faster than one at a time)
    print("Embedding chunks...")
    contents = [r["content"] for r in all_rows]
    embeddings = model.encode(contents, convert_to_tensor=False, show_progress_bar=True, batch_size=32)

    for row, emb in zip(all_rows, embeddings):
        row["embedding"] = emb.tolist()

    # Insert in batches
    print("Inserting into Supabase...")
    for i in range(0, len(all_rows), BATCH_SIZE):
        batch = all_rows[i:i + BATCH_SIZE]
        resp = supabase.table("code_chunks").insert(batch).execute()
        print(f"  Inserted rows {i} to {i+len(batch)}")

    print(f"\nDone. Ingested {len(all_rows)} chunks from {len(REPO_PATHS)} repo(s).")


if __name__ == "__main__":
    main()