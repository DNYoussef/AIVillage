import textwrap
from pathlib import Path

from monitoring.system_health_dashboard import ComponentHealthChecker


def test_legitimate_scores_higher_than_stub(tmp_path: Path) -> None:
    real_code = textwrap.dedent(
        """
        def add(a, b):
            return a + b
        """
    )
    stub_code = textwrap.dedent(
        """
        def add(a, b):
            return None
        """
    )

    real_file = tmp_path / "real.py"
    stub_file = tmp_path / "stub.py"
    real_file.write_text(real_code)
    stub_file.write_text(stub_code)

    checker = ComponentHealthChecker(project_root=tmp_path)
    checker._get_runtime_metrics = lambda: {"ping_success": 1.0, "error_rate": 0.0}  # type: ignore

    real_score = checker.check_component_health(real_file, "real")["implementation_score"]
    stub_score = checker.check_component_health(stub_file, "stub")["implementation_score"]

    assert real_score > stub_score
