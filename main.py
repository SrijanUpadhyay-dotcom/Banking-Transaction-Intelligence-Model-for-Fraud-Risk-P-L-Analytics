"""
BTI — Banking Transaction Intelligence Model
Production entry point.

Usage:
  # Start the REST API server
  python main.py api

  # Run the full pipeline (generate → clean → fraud rules → ML → P&L)
  python main.py pipeline [--force]

  # Initialise / migrate the database
  python main.py db-init [--seed]

  # Start the scheduler daemon
  python main.py scheduler

  # Run all tests
  python main.py test

  # Print system info
  python main.py info
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on the path so `bti` and `api` packages resolve
sys.path.insert(0, str(Path(__file__).parent))

# Install PyYAML if not present (needed by config module)
try:
    import yaml
except ImportError:
    os.system("pip install pyyaml -q")


def cmd_api(args):
    """Start the FastAPI server via uvicorn."""
    import uvicorn
    from bti.config import get_settings
    settings = get_settings()
    print(f"Starting BTI API v{settings.app_version} on {settings.api_host}:{settings.api_port}")
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=(settings.environment == "development"),
        log_level=settings.log_level.lower(),
    )


def cmd_pipeline(args):
    """Run the full analytics pipeline."""
    from bti.pipeline import run_full_pipeline
    from bti.logging_config import setup_logging
    from bti.config import get_settings
    settings = get_settings()
    setup_logging(settings.log_level)

    print(f"\nBTI Pipeline — {'FORCE RE-RUN' if args.force else 'INCREMENTAL'}")
    print("=" * 60)
    result = run_full_pipeline(force=args.force)

    print(f"\nRun ID  : {result['run_id']}")
    print(f"Status  : {result['status']}")
    print(f"Elapsed : {result.get('elapsed_seconds', '?')}s")
    print("\nStage results:")
    for stage, status in result.get("stages", {}).items():
        icon = "✓" if "OK" in str(status) or "SKIP" in str(status) else "✗"
        print(f"  {icon} {stage:<25} {status}")

    if result.get("db_rows_inserted"):
        print(f"\nDatabase: {result['db_rows_inserted']:,} rows inserted")

    if result["status"] == "FAILED":
        print(f"\nError: {result.get('error')}")
        sys.exit(1)


def cmd_db_init(args):
    """Create database tables and optionally seed from CSV."""
    from bti.database.init_db import init_database
    from bti.config import get_settings
    settings = get_settings()

    seed_path = None
    if args.seed:
        candidate = os.path.join(settings.processed_data_dir,
                                 "banking_transactions_ml_scored.csv")
        if Path(candidate).exists():
            seed_path = candidate
            print(f"Will seed from: {seed_path}")
        else:
            print(f"Seed file not found: {candidate}")
            print("Run 'python main.py pipeline' first to generate data.")

    print(f"Initialising database: {settings.database_url}")
    init_database(seed_csv=seed_path)
    print("Database ready.")


def cmd_scheduler(args):
    """Start the APScheduler pipeline daemon."""
    from scheduler.pipeline_scheduler import start_scheduler
    print("Starting BTI pipeline scheduler (Ctrl+C to stop)")
    start_scheduler()


def cmd_test(args):
    """Run the test suite."""
    import subprocess
    test_env = os.environ.copy()
    test_env.update({
        "BTI_DATABASE_URL": "sqlite:///:memory:",
        "BTI_SECRET_KEY": "test-secret-key-minimum-32-chars-long-for-tests",
        "BTI_API_KEY": "test-api-key-for-ci",
    })
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        env=test_env,
    )
    sys.exit(result.returncode)


def cmd_info(args):
    """Print system and configuration information."""
    from bti.config import get_settings
    settings = get_settings()

    print(f"\n{'='*60}")
    print(f"  Banking Transaction Intelligence Model v{settings.app_version}")
    print(f"{'='*60}")
    print(f"  Environment : {settings.environment}")
    print(f"  Database    : {settings.database_url[:60]}...")
    print(f"  Log Level   : {settings.log_level}")
    print(f"  Raw Data    : {settings.raw_data_path}")
    print(f"  Processed   : {settings.processed_data_dir}")
    print(f"  Outputs     : {settings.outputs_dir}")
    print(f"  Models      : {settings.models_dir}")
    print(f"  API         : http://{settings.api_host}:{settings.api_port}")
    print(f"  Alerts      : {'ENABLED' if settings.alerts_enabled else 'DISABLED'}")
    print()

    # Check what data exists
    print("  Data status:")
    checks = [
        ("Raw CSV", settings.raw_data_path),
        ("Cleaned CSV", os.path.join(settings.processed_data_dir, "banking_transactions_clean.csv")),
        ("Flagged CSV", os.path.join(settings.processed_data_dir, "banking_transactions_flagged.csv")),
        ("ML Scored CSV", os.path.join(settings.processed_data_dir, "banking_transactions_ml_scored.csv")),
        ("Excel Report", os.path.join(settings.outputs_dir, "banking_transaction_intelligence_model.xlsx")),
        ("PDF Report",   os.path.join(settings.outputs_dir, "banking_transaction_intelligence_report.pdf")),
    ]
    for name, path in checks:
        exists = Path(path).exists()
        size = f"({Path(path).stat().st_size // 1024}KB)" if exists else ""
        icon = "✓" if exists else "✗"
        print(f"    {icon} {name:<25} {size}")
    print()


def main():
    parser = argparse.ArgumentParser(
        prog="bti",
        description="Banking Transaction Intelligence Model — Production CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("api",       help="Start the FastAPI REST server")
    sub.add_parser("scheduler", help="Start the pipeline scheduler daemon")
    sub.add_parser("info",      help="Show system info and data status")
    sub.add_parser("test",      help="Run the test suite")

    p_pipe = sub.add_parser("pipeline", help="Run the full analytics pipeline")
    p_pipe.add_argument("--force", action="store_true",
                        help="Re-run all stages even if outputs exist")

    p_db = sub.add_parser("db-init", help="Initialise the database")
    p_db.add_argument("--seed", action="store_true",
                      help="Seed from processed CSV if available")

    args = parser.parse_args()

    dispatch = {
        "api":       cmd_api,
        "pipeline":  cmd_pipeline,
        "db-init":   cmd_db_init,
        "scheduler": cmd_scheduler,
        "test":      cmd_test,
        "info":      cmd_info,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
