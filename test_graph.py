from graph import build_graph

def main():
    graph = build_graph()
    query = "Give me a detailed research report on the latest quarterly earnings date announced for Netflix."
    result = graph.invoke({"messages": [], "query": query, "route": "", "result": "", "source_used": ""})
    print("ROUTE:", result.get("route"))
    print("RESULT:", result.get("result"))

if __name__ == "__main__":
    main()