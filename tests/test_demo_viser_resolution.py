import argparse
import ast
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loger.utils.image_resolution import normalize_target_resolution_arg, resolve_target_image_size


class DemoViserResolutionTests(unittest.TestCase):
    def test_demo_viser_cli_accepts_one_or_more_integer_resolution_values(self):
        source_path = Path(__file__).resolve().parents[1] / "demo_viser.py"
        source = source_path.read_text()
        tree = ast.parse(source)

        resolution_calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
                continue
            if not node.args or not isinstance(node.args[0], ast.Constant) or node.args[0].value != "--resolution":
                continue
            resolution_calls.append(node)

        self.assertEqual(len(resolution_calls), 1)
        call = resolution_calls[0]
        kwargs = {kw.arg: kw.value for kw in call.keywords}
        self.assertIsInstance(kwargs["type"], ast.Name)
        self.assertEqual(kwargs["type"].id, "int")
        self.assertEqual(kwargs["nargs"].value, "+")

    def test_resolution_single_value_parses_as_shorter_side(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--resolution", type=int, nargs="+", default=None)
        args = parser.parse_args(["--resolution", "280"])
        self.assertEqual(args.resolution, [280])
        self.assertEqual(normalize_target_resolution_arg(args.resolution), [280])

    def test_resolution_two_values_parse_as_explicit_width_height(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--resolution", type=int, nargs="+", default=None)
        args = parser.parse_args(["--resolution", "504", "280"])
        self.assertEqual(args.resolution, [504, 280])
        self.assertEqual(normalize_target_resolution_arg(args.resolution), [504, 280])

    def test_resolution_shorter_side_preserves_aspect_ratio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "frame.png"
            Image.new("RGB", (1920, 1080), color="white").save(image_path)

            target_size = resolve_target_image_size([str(image_path)], target_short_side=280)
            self.assertEqual(target_size, (504, 280))

    def test_resolution_rejects_invalid_arity(self):
        with self.assertRaisesRegex(ValueError, "expects one integer"):
            normalize_target_resolution_arg([504, 280, 140])


if __name__ == "__main__":
    unittest.main()
