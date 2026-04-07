import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEMO_VISER = ROOT / "demo_viser.py"
PI3_MODEL = ROOT / "loger" / "models" / "pi3.py"


def load_tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def find_function(tree: ast.AST, name: str):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Function {name!r} not found")


def collect_called_names(node: ast.AST) -> set[str]:
    names = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


def main() -> None:
    demo_tree = load_tree(DEMO_VISER)
    pi3_tree = load_tree(PI3_MODEL)

    forward_node = find_function(pi3_tree, "forward")
    consume_node = find_function(pi3_tree, "consume_stream_chunk")
    finalize_node = find_function(pi3_tree, "finalize_stream")
    run_stream_node = find_function(demo_tree, "run_streaming_inference")

    forward_calls = collect_called_names(forward_node)
    consume_calls = collect_called_names(consume_node)
    run_stream_calls = collect_called_names(run_stream_node)

    required_forward = {"init_stream_state", "consume_stream_chunk", "finalize_stream"}
    missing_forward = sorted(required_forward - forward_calls)
    if missing_forward:
        raise AssertionError(f"Pi3.forward missing streaming lifecycle calls: {missing_forward}")

    if "decode" not in consume_calls:
        raise AssertionError("consume_stream_chunk must still drive decode()")

    finalize_source = ast.get_source_segment(PI3_MODEL.read_text(), finalize_node) or ""
    if "Stream finalization called before all windows were consumed" not in finalize_source:
        raise AssertionError("finalize_stream must guard against partial consumption")

    required_driver_calls = {
        "get_window_ranges",
        "init_stream_state",
        "consume_stream_chunk",
        "finalize_stream",
        "empty_cache",
    }
    missing_driver = sorted(required_driver_calls - run_stream_calls)
    if missing_driver:
        raise AssertionError(f"run_streaming_inference missing expected calls: {missing_driver}")

    print("streaming-lifecycle: PASS")


if __name__ == "__main__":
    main()
