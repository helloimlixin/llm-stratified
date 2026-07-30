from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


def requirement_names(path: Path) -> set[str]:
    names: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        name = re.split(r"[<>=!~\[]", line, maxsplit=1)[0]
        names.add(name.strip().lower().replace("_", "-"))
    return names


class TestCIDependencies(unittest.TestCase):
    def test_ci_manifest_covers_direct_test_imports(self):
        names = requirement_names(ROOT / "requirements-ci.txt")
        self.assertTrue(
            {"hydra-core", "huggingface-hub", "matplotlib", "tqdm"}.issubset(names)
        )

    def test_workflow_installs_ci_manifest(self):
        workflow = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
        self.assertIn("python -m pip install -r requirements-ci.txt", workflow)


if __name__ == "__main__":
    unittest.main()
