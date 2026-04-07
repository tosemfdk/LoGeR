import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEMO_VISER = ROOT / "demo_viser.py"
PI3_MODEL = ROOT / "loger" / "models" / "pi3.py"


def load_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def function_names(tree: ast.AST) -> set[str]:
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def assert_contains(source: str, needle: str, label: str) -> None:
    if needle not in source:
        raise AssertionError(f"Missing expected snippet for {label!r}: {needle}")


def assert_not_contains(source: str, needle: str, label: str) -> None:
    if needle in source:
        raise AssertionError(f"Unexpected snippet for {label!r}: {needle}")


def main() -> None:
    demo_source = DEMO_VISER.read_text()
    pi3_source = PI3_MODEL.read_text()
    demo_tree = load_module(DEMO_VISER)
    pi3_tree = load_module(PI3_MODEL)

    demo_funcs = function_names(demo_tree)
    pi3_funcs = function_names(pi3_tree)

    expected_demo_funcs = {
        "collect_input_image_paths",
        "resolve_target_image_size",
        "load_image_chunk_from_paths",
        "iter_preprocessed_image_windows",
        "run_streaming_inference",
    }
    expected_pi3_funcs = {
        "get_window_ranges",
        "init_stream_state",
        "consume_stream_chunk",
        "finalize_stream",
    }

    missing_demo = sorted(expected_demo_funcs - demo_funcs)
    missing_pi3 = sorted(expected_pi3_funcs - pi3_funcs)
    if missing_demo:
        raise AssertionError(f"Missing demo streaming helpers: {missing_demo}")
    if missing_pi3:
        raise AssertionError(f"Missing Pi3 streaming methods: {missing_pi3}")

    assert_not_contains(demo_source, "def load_images_from_paths", "removed eager loader helper")
    assert_not_contains(
        demo_source,
        "load_images_from_paths(all_image_names_collected",
        "legacy eager load call removed from main inference path",
    )
    assert_contains(demo_source, "torch.cuda.empty_cache()", "per-chunk GPU cleanup")
    assert_contains(
        demo_source,
        "raw_model_predictions, _, _ = run_streaming_inference(",
        "main path uses streaming inference helper",
    )
    assert_contains(
        demo_source,
        "torch.save({k: torch.from_numpy(v) for k,v in predictions_dict.items()}, output_path)",
        "final .pt save still serializes merged prediction payload",
    )
    assert_contains(
        demo_source,
        "predictions_dict = torch.load(saved_predictions_path, map_location=\"cpu\", weights_only=False)",
        "existing .pt load path remains in place",
    )
    assert_contains(
        demo_source,
        "viser_wrapper(",
        "final merged payload still flows into viser_wrapper",
    )
    assert_contains(pi3_source, '"images"', "images included in Pi3 merge sequence keys")
    assert_contains(
        pi3_source,
        "Stream finalization called before all windows were consumed",
        "finalize_stream guard against partial consumption",
    )

    print("streaming-structure: PASS")


if __name__ == "__main__":
    main()
