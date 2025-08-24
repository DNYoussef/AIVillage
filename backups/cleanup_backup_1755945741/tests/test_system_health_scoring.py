from pathlib import Path
import textwrap

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


def test_todo_in_string_not_flagged(tmp_path: Path) -> None:
    code = textwrap.dedent(
        """
        def greet():
            return "This string mentions TODO but is fine"
        """
    )

    file = tmp_path / "greet.py"
    file.write_text(code)

    checker = ComponentHealthChecker(project_root=tmp_path)
    checker._get_runtime_metrics = lambda: {"ping_success": 1.0, "error_rate": 0.0}  # type: ignore

    result = checker.check_component_health(file, "greet")
    assert "TODO" not in result["stub_indicators"]
