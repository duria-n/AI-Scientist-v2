#!/usr/bin/env python3
"""Generate an AI Scientist idea from a topic and optionally launch the full pipeline."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from ai_scientist.llm import create_client, extract_json_between_markers, get_response_from_llm


DEFAULT_TEMPLATE_PATH = Path("templates/idea_from_topic.md.j2")
DEFAULT_OUTPUT_DIR = Path("ai_scientist/ideas/generated")


SYSTEM_PROMPT = """You are helping prepare an AI Scientist experiment specification.
Given a paper topic from the user, produce one concrete, feasible, high-quality research idea that can be executed by an autonomous research workflow.

Requirements:
- Return exactly one idea as JSON.
- Keep the idea ambitious but implementable in a normal academic lab.
- Favor simple, testable experiments over grand but vague plans.
- The topic may be in Chinese or English. Follow the user's language for prose fields.
- The "Name" field must be lowercase ASCII snake_case and short enough to use in filenames.
- The output must be self-contained and suitable for downstream experiment planning.
- Do not add citations or markdown outside the requested JSON block.

Required fields:
- "Name"
- "Title"
- "Keywords"
- "TLDR"
- "Short Hypothesis"
- "Related Work"
- "Abstract"
- "Experiments"
- "Risk Factors and Limitations"
"""


USER_PROMPT_TEMPLATE = """Generate one AI Scientist idea for the following paper topic.

Topic:
{topic}

Additional constraints:
{extra_instructions}

Return the result in this format:

```json
{{
  "idea": {{
    "Name": "...",
    "Title": "...",
    "Keywords": ["...", "..."],
    "TLDR": "...",
    "Short Hypothesis": "...",
    "Related Work": "...",
    "Abstract": "...",
    "Experiments": "...",
    "Risk Factors and Limitations": "..."
  }}
}}
```
"""


REQUIRED_FIELDS = [
    "Name",
    "Title",
    "Keywords",
    "TLDR",
    "Short Hypothesis",
    "Related Work",
    "Abstract",
    "Experiments",
    "Risk Factors and Limitations",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an idea from a paper topic and optionally run AI Scientist."
    )
    parser.add_argument(
        "--topic",
        help="Paper topic or research direction to expand into an AI Scientist idea.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for topic and common options interactively.",
    )
    parser.add_argument(
        "--model-idea",
        default="deepseek-chat",
        help="LLM model used to generate the idea JSON.",
    )
    parser.add_argument(
        "--template",
        default=str(DEFAULT_TEMPLATE_PATH),
        help="Path to the Jinja2 markdown template.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to store the generated idea JSON/MD files.",
    )
    parser.add_argument(
        "--extra-instructions",
        default=(
            "Prefer a topic-specific idea with clear evaluation metrics and a realistic first-stage implementation."
        ),
        help="Extra instruction appended to the idea-generation prompt.",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only generate the idea files; do not launch launch_scientist_bfts.py.",
    )
    parser.add_argument(
        "--writeup-type",
        default="icbinb",
        choices=["normal", "icbinb"],
        help="Writeup type to pass to launch_scientist_bfts.py.",
    )
    parser.add_argument(
        "--attempt-id",
        type=int,
        default=0,
        help="Attempt id passed through to launch_scientist_bfts.py.",
    )
    parser.add_argument(
        "--model-agg-plots",
        default="deepseek-chat",
        help="Plot aggregation model for the downstream launch script.",
    )
    parser.add_argument(
        "--model-writeup",
        default="deepseek-chat",
        help="Writeup model for the downstream launch script.",
    )
    parser.add_argument(
        "--model-writeup-small",
        default="deepseek-chat",
        help="Smaller writeup model for the downstream launch script.",
    )
    parser.add_argument(
        "--model-citation",
        default="deepseek-chat",
        help="Citation model for the downstream launch script.",
    )
    parser.add_argument(
        "--model-review",
        default="deepseek-chat",
        help="Review model for the downstream launch script.",
    )
    parser.add_argument(
        "--num-cite-rounds",
        type=int,
        default=8,
        help="Citation rounds for the downstream launch script.",
    )
    parser.add_argument(
        "--skip-writeup",
        action="store_true",
        help="Pass --skip_writeup to launch_scientist_bfts.py.",
    )
    parser.add_argument(
        "--run-review",
        action="store_true",
        help="Run the final review stage. By default review is skipped.",
    )
    parser.add_argument(
        "--add-dataset-ref",
        action="store_true",
        help="Pass --add_dataset_ref to launch_scientist_bfts.py.",
    )
    parser.add_argument(
        "--load-code",
        action="store_true",
        help="Pass --load_code to launch_scientist_bfts.py.",
    )
    return parser.parse_args()


def prompt_text(prompt: str, default: str | None = None, allow_empty: bool = False) -> str:
    while True:
        suffix = f" [{default}]" if default not in (None, "") else ""
        value = input(f"{prompt}{suffix}: ").strip()
        if value:
            return value
        if default is not None:
            return default
        if allow_empty:
            return ""
        print("This value is required.")


def prompt_bool(prompt: str, default: bool) -> bool:
    default_label = "Y/n" if default else "y/N"
    while True:
        value = input(f"{prompt} [{default_label}]: ").strip().lower()
        if not value:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Please answer y or n.")


def prompt_int(prompt: str, default: int) -> int:
    while True:
        value = input(f"{prompt} [{default}]: ").strip()
        if not value:
            return default
        try:
            return int(value)
        except ValueError:
            print("Please enter an integer.")


def prompt_choice(prompt: str, choices: list[str], default: str) -> str:
    choices_display = "/".join(choices)
    while True:
        value = input(f"{prompt} ({choices_display}) [{default}]: ").strip()
        if not value:
            return default
        if value in choices:
            return value
        print(f"Please choose one of: {', '.join(choices)}")


def populate_interactive_args(args: argparse.Namespace) -> argparse.Namespace:
    print("Interactive mode: topic -> idea generation -> optional AI Scientist launch")
    print()

    args.topic = prompt_text("Paper topic", args.topic)
    args.model_idea = prompt_text("Idea generation model", args.model_idea)
    args.template = prompt_text("Markdown template path", args.template)
    args.output_dir = prompt_text("Output directory", args.output_dir)
    args.extra_instructions = prompt_text(
        "Extra generation instructions",
        args.extra_instructions,
    )
    args.generate_only = prompt_bool("Only generate idea files", args.generate_only)

    if not args.generate_only:
        args.writeup_type = prompt_choice(
            "Writeup type",
            ["normal", "icbinb"],
            args.writeup_type,
        )
        args.attempt_id = prompt_int("Attempt id", args.attempt_id)

        shared_model = prompt_text(
            "Default downstream model",
            args.model_writeup,
        )
        configure_separately = prompt_bool(
            "Configure downstream models separately",
            False,
        )
        if configure_separately:
            args.model_agg_plots = prompt_text(
                "Model for plot aggregation",
                args.model_agg_plots,
            )
            args.model_writeup = prompt_text(
                "Model for writeup",
                args.model_writeup,
            )
            args.model_writeup_small = prompt_text(
                "Model for smaller writeup steps",
                args.model_writeup_small,
            )
            args.model_citation = prompt_text(
                "Model for citation collection",
                args.model_citation,
            )
            args.model_review = prompt_text(
                "Model for final review",
                args.model_review,
            )
        else:
            args.model_agg_plots = shared_model
            args.model_writeup = shared_model
            args.model_writeup_small = shared_model
            args.model_citation = shared_model
            args.model_review = shared_model

        args.num_cite_rounds = prompt_int(
            "Number of citation rounds",
            args.num_cite_rounds,
        )
        args.skip_writeup = prompt_bool("Skip writeup stage", args.skip_writeup)
        args.run_review = prompt_bool("Run final review stage", args.run_review)
        args.add_dataset_ref = prompt_bool(
            "Add dataset reference helper",
            args.add_dataset_ref,
        )
        args.load_code = prompt_bool(
            "Load companion code file if present",
            args.load_code,
        )

    print()
    print("Configuration summary:")
    summary_items = {
        "topic": args.topic,
        "model_idea": args.model_idea,
        "template": args.template,
        "output_dir": args.output_dir,
        "generate_only": args.generate_only,
    }
    if not args.generate_only:
        summary_items.update(
            {
                "writeup_type": args.writeup_type,
                "attempt_id": args.attempt_id,
                "model_agg_plots": args.model_agg_plots,
                "model_writeup": args.model_writeup,
                "model_writeup_small": args.model_writeup_small,
                "model_citation": args.model_citation,
                "model_review": args.model_review,
                "num_cite_rounds": args.num_cite_rounds,
                "skip_writeup": args.skip_writeup,
                "run_review": args.run_review,
                "add_dataset_ref": args.add_dataset_ref,
                "load_code": args.load_code,
            }
        )
    for key, value in summary_items.items():
        print(f"- {key}: {value}")

    if not prompt_bool("Continue with this configuration", True):
        raise SystemExit(0)

    return args


def slugify(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_text).strip("_").lower()
    slug = re.sub(r"_+", "_", slug)
    return slug or "generated_idea"


def ensure_name(idea: dict[str, Any], topic: str) -> None:
    name = str(idea.get("Name", "")).strip()
    name = slugify(name) if name else slugify(topic)
    idea["Name"] = name[:80]


def normalize_keywords(idea: dict[str, Any]) -> None:
    keywords = idea.get("Keywords", [])
    if isinstance(keywords, str):
        parts = [part.strip() for part in re.split(r"[,\n，；;]", keywords) if part.strip()]
        idea["Keywords"] = parts
    elif isinstance(keywords, list):
        idea["Keywords"] = [str(item).strip() for item in keywords if str(item).strip()]
    else:
        idea["Keywords"] = []


def validate_idea(idea: dict[str, Any], topic: str) -> dict[str, Any]:
    for field in REQUIRED_FIELDS:
        if field not in idea or idea[field] in (None, ""):
            raise ValueError(f"Generated idea is missing required field: {field}")

    ensure_name(idea, topic)
    normalize_keywords(idea)
    return idea


def generate_idea(topic: str, model: str, extra_instructions: str) -> dict[str, Any]:
    client, client_model = create_client(model)
    prompt = USER_PROMPT_TEMPLATE.format(
        topic=topic.strip(),
        extra_instructions=extra_instructions.strip(),
    )
    response_text, _ = get_response_from_llm(
        prompt=prompt,
        client=client,
        model=client_model,
        system_message=SYSTEM_PROMPT,
        print_debug=False,
    )
    json_output = extract_json_between_markers(response_text)
    if not json_output or "idea" not in json_output:
        raise ValueError(f"Failed to parse idea JSON from model output:\n{response_text}")
    return validate_idea(json_output["idea"], topic)


def render_markdown(template_path: Path, topic: str, idea: dict[str, Any]) -> str:
    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(template_path.name)
    return template.render(topic=topic, idea=idea)


def save_outputs(
    output_dir: Path,
    topic: str,
    idea: dict[str, Any],
    template_path: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = idea["Name"]
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"

    json_path.write_text(
        json.dumps([idea], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    md_path.write_text(
        render_markdown(template_path, topic, idea),
        encoding="utf-8",
    )
    return json_path, md_path


def build_launch_command(args: argparse.Namespace, idea_json_path: Path) -> list[str]:
    command = [
        sys.executable,
        "launch_scientist_bfts.py",
        "--writeup-type",
        args.writeup_type,
        "--load_ideas",
        str(idea_json_path),
        "--idea_idx",
        "0",
        "--attempt_id",
        str(args.attempt_id),
        "--model_agg_plots",
        args.model_agg_plots,
        "--model_writeup",
        args.model_writeup,
        "--model_writeup_small",
        args.model_writeup_small,
        "--model_citation",
        args.model_citation,
        "--model_review",
        args.model_review,
        "--num_cite_rounds",
        str(args.num_cite_rounds),
    ]

    if args.skip_writeup:
        command.append("--skip_writeup")
    if not args.run_review:
        command.append("--skip_review")
    if args.add_dataset_ref:
        command.append("--add_dataset_ref")
    if args.load_code:
        command.append("--load_code")

    return command


def main() -> int:
    args = parse_args()
    if args.interactive or not args.topic:
        if not sys.stdin.isatty():
            raise SystemExit("Interactive input requires a TTY. Pass --topic explicitly.")
        args = populate_interactive_args(args)

    if not args.topic:
        raise SystemExit("Missing required --topic. Use --interactive to enter it interactively.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    template_path = Path(args.template).expanduser().resolve()

    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    idea = generate_idea(args.topic, args.model_idea, args.extra_instructions)
    idea_json_path, idea_md_path = save_outputs(output_dir, args.topic, idea, template_path)

    print(f"Generated JSON: {idea_json_path}")
    print(f"Generated Markdown: {idea_md_path}")

    if args.generate_only:
        return 0

    command = build_launch_command(args, idea_json_path)
    print("Launching pipeline:")
    print(shlex.join(command))
    completed = subprocess.run(command, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
