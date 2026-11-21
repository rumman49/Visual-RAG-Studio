"""Story 1.5: hosted deployment manifests must agree with each other."""

import re
import tomllib
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
SHA = re.compile(r"ragspark(?:\.git)?@?[^\n]*?([0-9a-f]{40})")


def _text(name: str) -> str:
    return (BACKEND / name).read_text(encoding="utf-8")


def _pyproject_rev() -> str:
    data = tomllib.loads(_text("pyproject.toml"))
    return data["tool"]["uv"]["sources"]["ragspark"]["rev"]


def test_ragspark_is_pinned_to_one_commit_everywhere():
    rev = _pyproject_rev()
    assert re.fullmatch(r"[0-9a-f]{40}", rev)
    assert f"ragspark.git@{rev}" in _text("requirements.txt")
    assert rev in _text("uv.lock")


def test_space_entrypoint_dependencies_are_declared():
    requirements = _text("requirements.txt")
    assert re.search(r"^gradio[<>=]", requirements, re.MULTILINE)
    assert re.search(r"^uvicorn", requirements, re.MULTILINE)
    entry = _text("space_app.py")
    assert "import gradio" in entry and "import uvicorn" in entry


def test_dockerfile_uses_same_entrypoint_and_requirements():
    dockerfile = _text("Dockerfile")
    assert "pip install --no-cache-dir --user -r requirements.txt" in dockerfile
    assert 'CMD ["python", "space_app.py"]' in dockerfile
    assert "COPY --chown=user space_app.py" in dockerfile
    assert "7860" in dockerfile and "7860" in _text("space_app.py")


def test_space_front_matter_points_at_existing_entrypoint():
    readme = _text("README.md")
    match = re.search(r"^app_file:\s*(\S+)", readme, re.MULTILINE)
    assert match and (BACKEND / match.group(1)).exists()
