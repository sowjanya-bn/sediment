from langgraph.graph import END, StateGraph

from .nodes import (
    compute_penalty,
    finalize_record,
    handle_extraction_error,
    ingest_metadata,
    precheck_input,
    run_extract_initial,
    run_extract_strict,
    short_input_default_node,
    validate_initial,
    validate_strict,
    fallback_defaults,
)
from .routes import (
    route_after_initial_validation,
    route_after_precheck,
    route_after_strict_validation,
)
from .state import SedimentState


def build_sediment_graph():
    graph = StateGraph(SedimentState)

    graph.add_node("ingest_metadata", ingest_metadata)
    graph.add_node("precheck_input", precheck_input)
    graph.add_node("short_input_default", short_input_default_node)
    graph.add_node("compute_penalty", compute_penalty)
    graph.add_node("extract_initial", run_extract_initial)
    graph.add_node("validate_initial", validate_initial)
    graph.add_node("extract_strict", run_extract_strict)
    graph.add_node("validate_strict", validate_strict)
    graph.add_node("fallback_defaults", fallback_defaults)
    graph.add_node("handle_extraction_error", handle_extraction_error)
    graph.add_node("finalize_record", finalize_record)

    graph.set_entry_point("ingest_metadata")
    graph.add_edge("ingest_metadata", "precheck_input")

    graph.add_conditional_edges(
        "precheck_input",
        route_after_precheck,
        {
            "short": "short_input_default",
            "normal": "compute_penalty",
        },
    )

    graph.add_edge("short_input_default", END)

    graph.add_edge("compute_penalty", "extract_initial")

    graph.add_conditional_edges(
        "extract_initial",
        lambda state: "error" if state.get("extraction_error") else "ok",
        {
            "ok": "validate_initial",
            "error": "handle_extraction_error",
        },
    )

    graph.add_conditional_edges(
        "validate_initial",
        route_after_initial_validation,
        {
            "pass": "finalize_record",
            "retry": "extract_strict",
        },
    )

    graph.add_conditional_edges(
        "extract_strict",
        lambda state: "error" if state.get("extraction_error") else "ok",
        {
            "ok": "validate_strict",
            "error": "handle_extraction_error",
        },
    )

    graph.add_conditional_edges(
        "validate_strict",
        route_after_strict_validation,
        {
            "pass": "finalize_record",
            "fail": "fallback_defaults",
        },
    )

    graph.add_edge("fallback_defaults", "finalize_record")
    graph.add_edge("finalize_record", END)
    graph.add_edge("handle_extraction_error", "finalize_record")

    return graph.compile()