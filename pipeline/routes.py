from .state import SedimentState


def route_after_precheck(state: SedimentState) -> str:
    return "short" if state.get("is_short_or_flat") else "normal"


def route_after_initial_validation(state: SedimentState) -> str:
    return "pass" if state.get("validation_passed") else "retry"


def route_after_strict_validation(state: SedimentState) -> str:
    return "pass" if state.get("validation_passed") else "fail"