"""
Banking Transaction Intelligence Model
Master Execution Script — Run All Pipeline Stages in Sequence
Usage: python src/09_run_all.py
"""

import os
import sys
import time

sys.path.insert(0, ".")

STEPS = [
    ("01_data_generation",    "Generate 50,000 synthetic transactions"),
    ("02_data_cleaning",      "Data cleaning & validation"),
    ("03_fraud_rules_engine", "Apply 19 fraud detection rules"),
    ("04_ml_anomaly_detection","Machine learning anomaly detection"),
    ("05_pnl_analytics",      "FP&A & P&L analytics layer"),
    ("06_excel_workbook_builder", "Build Excel workbook"),
    ("07_pdf_report_generator",   "Generate PDF report"),
]

SKIP_IF_EXISTS = {
    "01_data_generation":    "data/raw/banking_transactions_raw.csv",
    "02_data_cleaning":      "data/processed/banking_transactions_clean.csv",
    "03_fraud_rules_engine": "data/processed/banking_transactions_flagged.csv",
    "04_ml_anomaly_detection":"data/processed/banking_transactions_ml_scored.csv",
}


def run_step(module_name, description, force=False):
    output_check = SKIP_IF_EXISTS.get(module_name)
    if not force and output_check and os.path.exists(output_check):
        print(f"  [SKIP] {description} — output already exists")
        return True

    print(f"\n{'='*60}")
    print(f"  STEP: {description}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        mod = __import__(f"src.{module_name}", fromlist=[""])
        # Re-run main block
        exec(open(f"src/{module_name}.py").read(), {"__name__": "__main__"})
        elapsed = time.time() - t0
        print(f"  ✓ Completed in {elapsed:.1f}s")
        return True
    except SystemExit:
        return True
    except Exception as e:
        print(f"  ✗ Error in {module_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main(force=False):
    print("\n" + "="*60)
    print("  BANKING TRANSACTION INTELLIGENCE MODEL")
    print("  Full Pipeline Execution")
    print("="*60)

    # Ensure directories exist
    for d in ["data/raw", "data/processed", "outputs/charts", "outputs/pnl", "outputs"]:
        os.makedirs(d, exist_ok=True)

    results = {}
    for module, desc in STEPS:
        success = run_step(module, desc, force=force)
        results[module] = success

    print("\n" + "="*60)
    print("  PIPELINE SUMMARY")
    print("="*60)
    for module, desc in STEPS:
        status = "✓" if results.get(module) else "✗"
        print(f"  {status}  {desc}")

    print("\n  Output files:")
    outputs = [
        ("data/processed/banking_transactions_ml_scored.csv", "Full scored dataset"),
        ("outputs/banking_transaction_intelligence_model.xlsx", "Excel workbook"),
        ("outputs/banking_transaction_intelligence_report.pdf", "PDF report"),
        ("outputs/charts/eda_overview.png",          "EDA charts"),
        ("outputs/charts/pnl_dashboard.png",         "P&L charts"),
        ("outputs/charts/rf_feature_importance.png", "RF feature importance"),
        ("outputs/pnl/monthly_variance.csv",         "Monthly P&L variance"),
        ("outputs/pnl/exception_queue.csv",          "Exception queue"),
    ]
    for path, desc in outputs:
        exists = "✓" if os.path.exists(path) else "○"
        print(f"  {exists}  {path:<55} {desc}")

    print("\n  Run Streamlit dashboard:")
    print("  $ streamlit run dashboard/streamlit_bti_dashboard.py")
    print("\n  Open HTML terminal:")
    print("  $ open dashboard/bti_terminal_dashboard.html")
    print()


if __name__ == "__main__":
    force = "--force" in sys.argv
    main(force=force)
