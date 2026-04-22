from pipeline import build_sediment_graph

_GRAPH = None

def _get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_sediment_graph()
    return _GRAPH

def extract(raw_text: str, backend: str = "anthropic", ollama_model: str = "llama3.2") -> dict:
    graph = _get_graph()
    state = graph.invoke(
        {
            "raw_text": raw_text,
            "backend": backend,
            "ollama_model": ollama_model,
        }
    )
    return state["record"]