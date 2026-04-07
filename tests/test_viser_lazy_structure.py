import ast
import unittest
from pathlib import Path


class ViserLazyStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_path = Path(__file__).resolve().parents[1] / "loger" / "utils" / "viser_utils.py"
        cls.source = cls.source_path.read_text()
        cls.tree = ast.parse(cls.source)

    def test_viser_wrapper_defines_lazy_scene_helpers(self):
        viser_wrapper = next(
            node for node in self.tree.body if isinstance(node, ast.FunctionDef) and node.name == "viser_wrapper"
        )
        nested_names = {
            node.name
            for node in ast.walk(viser_wrapper)
            if isinstance(node, ast.FunctionDef)
        }
        self.assertTrue(
            {"create_timestep_handles", "remove_timestep_handles", "compute_visible_timesteps", "set_visibility"}.issubset(
                nested_names
            )
        )

    def test_viser_wrapper_no_longer_eagerly_builds_all_frames_with_tqdm_loop(self):
        self.assertNotIn("for i in tqdm(range(S))", self.source)
        self.assertIn("Initializing lazy viser scene for current visible frames", self.source)

    def test_lazy_scene_removes_frame_root_handles(self):
        self.assertIn("frame_handle.remove()", self.source)

    def test_subsample_reload_forces_lazy_scene_rebuild(self):
        self.assertIn("set_visibility(force_reload=True)", self.source)


if __name__ == "__main__":
    unittest.main()
