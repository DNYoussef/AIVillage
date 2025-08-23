import subprocess


def test_no_pickle_loads():
    result = subprocess.run(
        ["git", "grep", "-n", "pickle\\.loads", "src"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.stdout == ""
