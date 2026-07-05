"""
Pipeline orchestrator — runs each stage in order, tracks run metadata,
persists results to the database, and dispatches alerts on completion.

Each stage is idempotent: if the output already exists and force=False,
the stage is skipped. This allows safe re-runs after partial failures.
"""

import importlib.util
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from bti.config import get_settings
from bti.logging_config import get_logger, AuditLogger
from bti.alerts import AlertDispatcher

log = get_logger("pipeline")
settings = get_settings()


def _dynamic_import(name: str, filepath: str):
    """Load a module from a file path, bypassing Python's import system."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class PipelineStage:
    def __init__(self, name: str, script: str, output_marker: str):
        self.name = name
        self.script = str(Path(settings.models_dir).parent / script)
        self.output_marker = output_marker

    @property
    def output_exists(self) -> bool:
        return Path(self.output_marker).exists()


STAGES = [
    PipelineStage("data_generation",   "src/01_data_generation.py",
                  settings.raw_data_path),
    PipelineStage("data_cleaning",     "src/02_data_cleaning.py",
                  os.path.join(settings.processed_data_dir, "banking_transactions_clean.csv")),
    PipelineStage("fraud_rules",       "src/03_fraud_rules_engine.py",
                  os.path.join(settings.processed_data_dir, "banking_transactions_flagged.csv")),
    PipelineStage("ml_detection",      "src/04_ml_anomaly_detection.py",
                  os.path.join(settings.processed_data_dir, "banking_transactions_ml_scored.csv")),
    PipelineStage("pnl_analytics",     "src/05_pnl_analytics.py",
                  os.path.join(settings.outputs_dir, "pnl", "monthly_variance.csv")),
]


class PipelineOrchestrator:
    """
    Manages a full pipeline run end-to-end:
      1. Data generation  →  2. Cleaning  →  3. Fraud rules
      →  4. ML detection  →  5. P&L analytics
      →  6. Database persistence  →  7. Alert dispatch
    """

    def __init__(self, run_id: Optional[str] = None, force: bool = False):
        self.run_id = run_id or str(uuid.uuid4())[:8]
        self.force = force
        self.audit = AuditLogger(settings.audit_log_path)
        self.dispatcher = AlertDispatcher()
        self._start_ts: Optional[float] = None

    def run(self) -> dict:
        self._start_ts = time.time()
        log.info("Pipeline run started", extra={"run_id": self.run_id, "force": self.force})
        self.audit.record("PIPELINE_START", "N/A", {"run_id": self.run_id})

        results = {"run_id": self.run_id, "stages": {}, "status": "SUCCESS"}

        try:
            for stage in STAGES:
                self._run_stage(stage, results)

            self._persist_to_db(results)
            self._dispatch_alerts()

        except Exception as exc:
            results["status"] = "FAILED"
            results["error"] = str(exc)
            log.exception("Pipeline failed", extra={"run_id": self.run_id})
            self.audit.record("PIPELINE_FAIL", "N/A", {"run_id": self.run_id, "error": str(exc)})
        finally:
            elapsed = round(time.time() - self._start_ts, 1)
            results["elapsed_seconds"] = elapsed
            log.info("Pipeline run finished",
                     extra={"run_id": self.run_id, "status": results["status"],
                             "elapsed_s": elapsed})
            self.audit.record("PIPELINE_END", "N/A",
                              {"run_id": self.run_id, "status": results["status"],
                               "elapsed_s": elapsed})
        return results

    def _run_stage(self, stage: PipelineStage, results: dict) -> None:
        if not self.force and stage.output_exists:
            log.info(f"[SKIP] {stage.name} — output exists")
            results["stages"][stage.name] = "SKIPPED"
            return

        log.info(f"[START] {stage.name}")
        t0 = time.time()
        try:
            script_path = stage.script
            if not Path(script_path).exists():
                raise FileNotFoundError(f"Script not found: {script_path}")
            # Execute via exec() with __name__ == "__main__" so each script's
            # if __name__ == "__main__": block runs exactly once.
            orig_dir = os.getcwd()
            project_root = str(Path(script_path).parent.parent)
            os.chdir(project_root)
            try:
                _exec_script(script_path)
            finally:
                os.chdir(orig_dir)
            elapsed = round(time.time() - t0, 1)
            log.info(f"[DONE] {stage.name}", extra={"elapsed_s": elapsed})
            results["stages"][stage.name] = f"OK ({elapsed}s)"
        except Exception as exc:
            log.exception(f"[FAIL] {stage.name}")
            results["stages"][stage.name] = f"FAILED: {exc}"
            raise

    def _persist_to_db(self, results: dict) -> None:
        ml_csv = os.path.join(settings.processed_data_dir, "banking_transactions_ml_scored.csv")
        if not Path(ml_csv).exists():
            log.warning("ML CSV not found — skipping database persistence")
            return

        try:
            from bti.database.init_db import seed_from_csv
            inserted = seed_from_csv(ml_csv)
            results["db_rows_inserted"] = inserted
            log.info("Database persistence complete", extra={"rows": inserted})
        except Exception as exc:
            log.exception("Database persistence failed — data still available in CSVs")
            results["db_rows_inserted"] = 0
            results["db_error"] = str(exc)

    def _dispatch_alerts(self) -> None:
        ml_csv = os.path.join(settings.processed_data_dir, "banking_transactions_ml_scored.csv")
        if not Path(ml_csv).exists():
            return
        try:
            df = pd.read_csv(ml_csv, usecols=["transaction_id", "customer_id", "transaction_amount",
                                                "channel", "final_risk_score", "final_alert_tier"],
                             low_memory=False)
            alerts = df[df["final_alert_tier"].isin(["CRITICAL", "VERY HIGH"])].to_dict(orient="records")
            self.dispatcher.dispatch(alerts)
        except Exception:
            log.exception("Alert dispatch failed")


def _exec_script(path: str) -> None:
    """Execute a Python script file in its own global namespace."""
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()
    ns = {"__name__": "__main__", "__file__": path}
    exec(compile(code, path, "exec"), ns)


def run_full_pipeline(force: bool = False) -> dict:
    """Convenience function — use from the CLI or scheduler."""
    return PipelineOrchestrator(force=force).run()
