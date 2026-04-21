import json
import tempfile
import unittest
from pathlib import Path

from ai_scientist.perform_icbinb_writeup import (
    filter_experiment_summaries,
    load_exp_summaries,
)


class SummaryPostprocessingTests(unittest.TestCase):
    def test_filter_experiment_summaries_handles_null_stage_payloads(self):
        exp_summaries = {
            "BASELINE_SUMMARY": None,
            "RESEARCH_SUMMARY": {
                "best node": {
                    "overall_plan": "useful plan",
                    "analysis": "useful analysis",
                    "metric": {"value": 1.0},
                    "code": "print('x')",
                    "ignored": "should be dropped",
                }
            },
            "ABLATION_SUMMARY": None,
        }

        filtered = filter_experiment_summaries(
            exp_summaries, step_name="plot_aggregation"
        )

        self.assertEqual(filtered["BASELINE_SUMMARY"], {})
        self.assertEqual(filtered["ABLATION_SUMMARY"], {})
        self.assertEqual(
            filtered["RESEARCH_SUMMARY"]["best node"],
            {
                "overall_plan": "useful plan",
                "analysis": "useful analysis",
            },
        )

    def test_load_exp_summaries_normalizes_null_json_to_empty_structures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            logs_dir = base / "logs" / "0-run"
            logs_dir.mkdir(parents=True, exist_ok=True)

            (logs_dir / "baseline_summary.json").write_text("null")
            (logs_dir / "research_summary.json").write_text(
                json.dumps({"best node": {"analysis": "ok"}})
            )
            (logs_dir / "ablation_summary.json").write_text("null")

            loaded = load_exp_summaries(str(base))

        self.assertEqual(loaded["BASELINE_SUMMARY"], {})
        self.assertEqual(loaded["RESEARCH_SUMMARY"], {"best node": {"analysis": "ok"}})
        self.assertEqual(loaded["ABLATION_SUMMARY"], [])

    def test_filter_experiment_summaries_handles_null_best_node(self):
        exp_summaries = {
            "BASELINE_SUMMARY": {"best node": None},
            "RESEARCH_SUMMARY": None,
            "ABLATION_SUMMARY": [],
        }

        filtered = filter_experiment_summaries(exp_summaries, step_name="writeup")

        self.assertEqual(filtered["BASELINE_SUMMARY"], {})
        self.assertEqual(filtered["RESEARCH_SUMMARY"], {})
        self.assertEqual(filtered["ABLATION_SUMMARY"], {})

    def test_filter_experiment_summaries_includes_ablations_for_writeup(self):
        exp_summaries = {
            "BASELINE_SUMMARY": {},
            "RESEARCH_SUMMARY": {},
            "ABLATION_SUMMARY": [
                {
                    "ablation_name": "drop evidence linker",
                    "overall_plan": "remove module",
                    "analysis": "performance drops",
                    "metric": {"value": 0.7},
                    "plot_analyses": ["plot note"],
                    "ignored": "drop me",
                }
            ],
        }

        filtered = filter_experiment_summaries(exp_summaries, step_name="writeup")

        self.assertEqual(
            filtered["ABLATION_SUMMARY"],
            {
                "drop evidence linker": {
                    "overall_plan": "remove module",
                    "analysis": "performance drops",
                    "metric": {"value": 0.7},
                    "plot_analyses": ["plot note"],
                }
            },
        )


if __name__ == "__main__":
    unittest.main()
