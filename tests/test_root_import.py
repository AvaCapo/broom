import subprocess
import sys


def test_root_import_does_not_eagerly_import_optional_algorithm_dependencies() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import broom; "
            "assert 'scipy' not in sys.modules; "
            "assert 'torch' not in sys.modules; "
            "assert broom.__all__ == ('Hierarchy', 'Joint', 'Motion')",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
