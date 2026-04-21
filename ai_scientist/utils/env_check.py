from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_PYTHON = (3, 11)

# Map import names to the distribution name users should install.
DEFAULT_REQUIRED_MODULES = {
    "anthropic": "anthropic",
    "backoff": "backoff",
    "boto3": "boto3",
    "botocore": "botocore",
    "coolname": "coolname",
    "dataclasses_json": "dataclasses-json",
    "datasets": "datasets",
    "funcy": "funcy",
    "genson": "genson",
    "humanize": "humanize",
    "igraph": "python-igraph",
    "jinja2": "jinja2",
    "jsonschema": "jsonschema",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "omegaconf": "omegaconf",
    "openai": "openai",
    "pandas": "pandas",
    "PIL": "pillow",
    "psutil": "psutil",
    "pymupdf4llm": "pymupdf4llm",
    "pypdf": "pypdf",
    "requests": "requests",
    "rich": "rich",
    "shutup": "shutup",
    "tiktoken": "tiktoken",
    "torch": "torch",
    "tqdm": "tqdm",
    "transformers": "transformers",
    "wandb": "wandb",
    "yaml": "PyYAML",
}

DEFAULT_OPTIONAL_MODULES = {
    "IPython": "Optional. Needed only when exec.format_tb_ipython=True.",
}


@dataclass
class EnvironmentCheckResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _find_missing_modules(module_map: dict[str, str]) -> dict[str, str]:
    return {
        module_name: package_name
        for module_name, package_name in module_map.items()
        if importlib.util.find_spec(module_name) is None
    }


def _run_pip_check() -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return False, f"Unable to run `pip check`: {exc}"

    output = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    ).strip()

    if completed.returncode == 0:
        return True, output or "No broken requirements found."

    return False, output or "`pip check` reported dependency problems."


def check_runtime_environment(
    *,
    supported_python: tuple[int, int] = SUPPORTED_PYTHON,
    required_modules: dict[str, str] | None = None,
    optional_modules: dict[str, str] | None = None,
) -> EnvironmentCheckResult:
    required_modules = required_modules or DEFAULT_REQUIRED_MODULES
    optional_modules = optional_modules or DEFAULT_OPTIONAL_MODULES

    result = EnvironmentCheckResult()

    if sys.version_info[:2] != supported_python:
        expected = ".".join(str(part) for part in supported_python)
        current = ".".join(str(part) for part in sys.version_info[:3])
        result.errors.append(
            f"Unsupported Python interpreter: expected Python {expected}, got Python {current} at {sys.executable}."
        )

    missing_required = _find_missing_modules(required_modules)
    if missing_required:
        missing_packages = ", ".join(
            sorted(
                {
                    package_name
                    for package_name in missing_required.values()
                    if package_name
                }
            )
        )
        result.errors.append(
            "Missing required Python packages: "
            f"{missing_packages}. Install them with `python -m pip install -r requirements.txt`."
        )

    missing_optional = _find_missing_modules(optional_modules)
    for module_name, note in sorted(missing_optional.items()):
        result.warnings.append(f"{module_name} is not installed. {note}")

    pip_ok, pip_message = _run_pip_check()
    if not pip_ok:
        result.errors.append(pip_message)

    return result


def format_environment_check_report(
    result: EnvironmentCheckResult,
    *,
    env_name: str = "ai_scientist",
    repo_root: Path | None = None,
) -> str:
    repo_root = Path(repo_root) if repo_root is not None else Path.cwd()
    lines = []

    if result.errors:
        lines.append("Environment preflight failed:")
        lines.extend(f"- {message}" for message in result.errors)
        lines.append(
            f"- Recommended setup: `conda activate {env_name}` then `python -m pip install -r {repo_root / 'requirements.txt'}`."
        )
    else:
        lines.append("Environment preflight passed.")

    if result.warnings:
        lines.append("Warnings:")
        lines.extend(f"- {message}" for message in result.warnings)

    return "\n".join(lines)


def validate_runtime_environment(
    *,
    env_name: str = "ai_scientist",
    repo_root: Path | None = None,
) -> None:
    result = check_runtime_environment()
    if result.ok:
        return

    raise RuntimeError(
        format_environment_check_report(
            result,
            env_name=env_name,
            repo_root=repo_root,
        )
    )


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    result = check_runtime_environment()
    print(
        format_environment_check_report(
            result,
            env_name="ai_scientist",
            repo_root=repo_root,
        )
    )
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
