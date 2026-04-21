import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ai_scientist.treesearch.agent_manager import AgentManager, Stage
from ai_scientist.treesearch.journal import Journal, Node
from ai_scientist.treesearch.log_summarization import get_stage_summary
from ai_scientist.treesearch.parallel_agent import (
    CODE_GENERATION_FAILURE_SCRIPT,
    _apply_metric_validation,
    _run_code_query,
)
from ai_scientist.treesearch.utils.config import save_run
from ai_scientist.treesearch.utils.metric import MetricValue, validate_metric_value
from ai_scientist.treesearch.utils.response import extract_code


def make_metric(metric_name: str, final_value: float, best_value: float | None = None):
    if best_value is None:
        best_value = final_value
    return MetricValue(
        value={
            "metric_names": [
                {
                    "metric_name": metric_name,
                    "lower_is_better": "loss" in metric_name.lower(),
                    "description": metric_name,
                    "data": [
                        {
                            "dataset_name": "synthetic_dataset",
                            "final_value": final_value,
                            "best_value": best_value,
                        }
                    ],
                }
            ]
        }
    )


class JournalCandidateTests(unittest.TestCase):
    def test_node_classification_hybrid_plot_policy(self):
        journal = Journal()
        runnable_unreviewed = Node(
            plan="good",
            code="print('good')",
            metric=make_metric("evidence traceability score", 100.0),
            is_buggy=False,
            is_buggy_plots=None,
        )
        runnable_plot_validated = Node(
            plan="great",
            code="print('great')",
            metric=make_metric("validation loss", 0.1),
            is_buggy=False,
            is_buggy_plots=False,
        )
        runnable_plot_rejected = Node(
            plan="bad-plot",
            code="print('bad-plot')",
            metric=make_metric("validation loss", 0.2),
            is_buggy=False,
            is_buggy_plots=True,
        )
        buggy = Node(plan="buggy", code="raise RuntimeError()", is_buggy=True)

        for node in (
            runnable_unreviewed,
            runnable_plot_validated,
            runnable_plot_rejected,
            buggy,
        ):
            journal.append(node)

        self.assertEqual(
            {node.id for node in journal.runnable_nodes},
            {
                runnable_unreviewed.id,
                runnable_plot_validated.id,
                runnable_plot_rejected.id,
            },
        )
        self.assertEqual(
            {node.id for node in journal.good_nodes},
            {runnable_unreviewed.id, runnable_plot_validated.id},
        )
        self.assertEqual(
            {node.id for node in journal.plot_validated_good_nodes},
            {runnable_plot_validated.id},
        )


class StageCompletionTests(unittest.TestCase):
    def test_stage1_completes_with_good_node_even_without_plot_review(self):
        stage = Stage(
            name="1_initial_implementation_1_preliminary",
            description="preliminary",
            goals=[],
            max_iterations=20,
            num_drafts=3,
            stage_number=1,
        )
        journal = Journal(
            nodes=[
                Node(
                    step=0,
                    plan="good",
                    code="print('ok')",
                    metric=make_metric("evidence traceability score", 100.0),
                    is_buggy=False,
                    is_buggy_plots=None,
                )
            ]
        )
        manager = AgentManager.__new__(AgentManager)
        manager.journals = {stage.name: journal}
        manager.cfg = SimpleNamespace(
            exec=SimpleNamespace(timeout=3600),
            agent=SimpleNamespace(stages=SimpleNamespace(stage3_max_iters=12)),
        )
        manager.current_stage = stage

        is_complete, reason = AgentManager._check_stage_completion(manager, stage)

        self.assertTrue(is_complete)
        self.assertEqual(reason, "Found working implementation")


class CodeExtractionTests(unittest.TestCase):
    def test_extract_code_rejects_natural_language_prefixed_completion(self):
        completion = "I'll implement the fix below.\nprint('hello world')"
        self.assertEqual(extract_code(completion), "")

    def test_code_query_falls_back_to_deterministic_failure_script(self):
        with patch(
            "ai_scientist.treesearch.parallel_agent.query",
            return_value="I will now describe the code without a fenced block.",
        ):
            code = _run_code_query(
                {"Instructions": {}}, model="deepseek-chat", temperature=0.0, retries=2
            )

        self.assertEqual(code, CODE_GENERATION_FAILURE_SCRIPT)


class MetricValidationTests(unittest.TestCase):
    def test_invalid_traceability_marks_node_buggy(self):
        node = Node(
            plan="metric-bug",
            code="print('metric-bug')",
            metric=make_metric("evidence traceability score", 136.67),
            analysis="LLM reviewer said this looked fine.",
            is_buggy=False,
        )

        _apply_metric_validation(node)

        self.assertTrue(node.is_buggy)
        self.assertIn("Invalid parsed metrics", node.analysis)
        self.assertIsNone(node.metric.value)

    def test_metric_validation_accepts_valid_score_and_rejects_negative_or_nonfinite(self):
        valid_traceability = make_metric("evidence traceability score", 100.0)
        negative_loss = make_metric("validation loss", -0.1)
        nan_loss = make_metric("validation loss", float("nan"))

        self.assertEqual(validate_metric_value(valid_traceability), [])
        self.assertTrue(validate_metric_value(negative_loss))
        self.assertTrue(validate_metric_value(nan_loss))


class SummaryAndArtifactTests(unittest.TestCase):
    def test_stage_summary_returns_none_without_candidates(self):
        journal = Journal(nodes=[Node(step=0, plan="buggy", code="raise", is_buggy=True)])

        with patch(
            "ai_scientist.treesearch.log_summarization.get_response_from_llm"
        ) as mocked_llm:
            summary = get_stage_summary(
                journal,
                "1_initial_implementation_1_preliminary",
                model="deepseek-chat",
                client=None,
                research_context="Title: Generic Topic\nShort Hypothesis: Generic Test Hypothesis",
            )

        self.assertIsNone(summary)
        mocked_llm.assert_not_called()

    def test_save_run_exports_best_solution_using_stage_strategy(self):
        journal = Journal()
        good_node = Node(
            plan="good",
            code="print('good')\n",
            metric=make_metric("evidence traceability score", 95.0),
            is_buggy=False,
            is_buggy_plots=None,
        )
        buggy_node = Node(
            plan="buggy",
            code="print('buggy')\n",
            metric=make_metric("evidence traceability score", 99.0),
            is_buggy=True,
        )
        journal.append(good_node)
        journal.append(buggy_node)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = SimpleNamespace(log_dir=Path(tmpdir), exp_name="run")
            with patch(
                "ai_scientist.treesearch.utils.config.serialize.dump_json"
            ), patch(
                "ai_scientist.treesearch.utils.config.OmegaConf.save"
            ), patch(
                "ai_scientist.treesearch.utils.config.tree_export.generate"
            ):
                save_run(cfg, journal, stage_name="stage_1_initial_implementation")

            save_dir = Path(tmpdir) / "stage_1_initial_implementation"
            best_solution_path = save_dir / f"best_solution_{good_node.id}.py"
            best_node_id_path = save_dir / "best_node_id.txt"

            self.assertTrue(best_solution_path.exists())
            self.assertEqual(best_solution_path.read_text(), good_node.code)
            self.assertEqual(best_node_id_path.read_text(), good_node.id)


if __name__ == "__main__":
    unittest.main()
