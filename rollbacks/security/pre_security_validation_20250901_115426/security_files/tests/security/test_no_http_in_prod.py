import subprocess


def test_no_http_in_prod():
    result = subprocess.run(
        ["git", "grep", "-n", "http://", "src/production"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.stdout == ""
