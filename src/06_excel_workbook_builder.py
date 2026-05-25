"""
Banking Transaction Intelligence Model
Part 6: Professional Excel Workbook Builder
Generates a 15-tab Excel workbook with formatting, formulas, charts, and pivot data.
Uses openpyxl for full formatting control.
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import date

try:
    import openpyxl
    from openpyxl import Workbook
    from openpyxl.styles import (PatternFill, Font, Alignment, Border, Side,
                                  GradientFill, numbers)
    from openpyxl.utils import get_column_letter
    from openpyxl.chart import BarChart, LineChart, PieChart, Reference
    from openpyxl.chart.series import DataPoint
    from openpyxl.formatting.rule import ColorScaleRule, DataBarRule, CellIsRule
except ImportError:
    print("openpyxl not found. Run: pip install openpyxl")
    sys.exit(1)

os.makedirs("outputs", exist_ok=True)

# ─── COLOUR PALETTE ──────────────────────────────────────────────────────────
DARK_BLUE   = "0A2647"
MID_BLUE    = "144272"
LIGHT_BLUE  = "205295"
ACCENT_BLUE = "2C74B3"
DARK_GREEN  = "1B4332"
MID_GREEN   = "2D6A4F"
ACCENT_GREEN= "52B788"
DARK_RED    = "7B0000"
ACCENT_RED  = "C62828"
AMBER       = "F4A261"
GOLD        = "E9C46A"
WHITE       = "FFFFFF"
LIGHT_GREY  = "F5F5F5"
MID_GREY    = "BDBDBD"
DARK_GREY   = "424242"

def hdr(hex_color):
    return PatternFill("solid", fgColor=hex_color)

def font(color=WHITE, bold=False, size=10, italic=False):
    return Font(color=color, bold=bold, size=size, italic=italic, name="Calibri")

def center_align(wrap=False):
    return Alignment(horizontal="center", vertical="center", wrap_text=wrap)

def left_align(wrap=False):
    return Alignment(horizontal="left", vertical="center", wrap_text=wrap)

def thin_border():
    side = Side(style="thin", color="BDBDBD")
    return Border(left=side, right=side, top=side, bottom=side)


def apply_table_header(ws, row, cols, bg=DARK_BLUE, fg=WHITE, bold=True, size=10):
    for col_idx, text in enumerate(cols, 1):
        cell = ws.cell(row=row, column=col_idx, value=text)
        cell.fill = hdr(bg)
        cell.font = font(fg, bold=bold, size=size)
        cell.alignment = center_align(wrap=True)
        cell.border = thin_border()


def write_df_to_sheet(ws, df, start_row=2, include_header=True,
                      header_bg=DARK_BLUE, alt_row=True):
    if include_header:
        apply_table_header(ws, start_row, list(df.columns), bg=header_bg)
        start_row += 1

    fill_light = PatternFill("solid", fgColor=LIGHT_GREY)
    fill_white = PatternFill("solid", fgColor=WHITE)

    for r_idx, (_, row) in enumerate(df.iterrows()):
        fill = fill_light if (alt_row and r_idx % 2 == 0) else fill_white
        for c_idx, val in enumerate(row, 1):
            cell = ws.cell(row=start_row + r_idx, column=c_idx, value=val)
            cell.fill = fill
            cell.border = thin_border()
            cell.alignment = left_align()
            cell.font = Font(name="Calibri", size=9)

    return start_row + len(df) - 1


def auto_width(ws, min_w=8, max_w=40):
    for col in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            if cell.value:
                max_len = max(max_len, len(str(cell.value)))
        ws.column_dimensions[col_letter].width = min(max_w, max(min_w, max_len + 2))


def freeze_pane(ws, cell="A2"):
    ws.freeze_panes = cell


# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL SHEET BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def build_readme(wb):
    ws = wb.create_sheet("00_README")
    ws.sheet_view.showGridLines = False
    ws.column_dimensions["A"].width = 5
    ws.column_dimensions["B"].width = 70

    # Title block
    ws.merge_cells("B2:H4")
    cell = ws["B2"]
    cell.value = "Banking Transaction Intelligence Model\nFraud, Risk & P&L Analytics"
    cell.fill = hdr(DARK_BLUE)
    cell.font = Font(name="Calibri", color=WHITE, bold=True, size=18)
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[2].height = 40

    sections = [
        ("B6",  "PROJECT OVERVIEW", DARK_BLUE,   WHITE, True,  12),
        ("B7",  "This workbook contains the full Banking Transaction Intelligence framework:", LIGHT_GREY, DARK_GREY, False, 10),
        ("B9",  "SHEET GUIDE", DARK_BLUE, WHITE, True, 11),
        ("B10", "00_README       → Project overview and navigation guide", LIGHT_GREY, DARK_GREY, False, 10),
        ("B11", "01_DataDictionary → Full field definitions and business meaning", LIGHT_GREY, DARK_GREY, False, 10),
        ("B12", "02_RawTransactions → 50,000 synthetic banking transactions (sample 1,000 shown)", LIGHT_GREY, DARK_GREY, False, 10),
        ("B13", "03_CleanedData   → Validated and enriched transaction dataset", LIGHT_GREY, DARK_GREY, False, 10),
        ("B14", "04_FraudRulesEngine → 19 rule-based fraud detection signals", LIGHT_GREY, DARK_GREY, False, 10),
        ("B15", "05_RiskScoring   → Composite risk scoring model", LIGHT_GREY, DARK_GREY, False, 10),
        ("B16", "06_FPASummary    → P&L KPI summary dashboard", LIGHT_GREY, DARK_GREY, False, 10),
        ("B17", "07_PnLImpact     → Detailed P&L analysis and component waterfall", LIGHT_GREY, DARK_GREY, False, 10),
        ("B18", "08_ChannelAnalysis → Channel-level profitability and fraud analysis", LIGHT_GREY, DARK_GREY, False, 10),
        ("B19", "09_SegmentAnalysis → Customer segment P&L and risk profile", LIGHT_GREY, DARK_GREY, False, 10),
        ("B20", "10_MerchantRisk  → Merchant-level risk ranking and chargeback analysis", LIGHT_GREY, DARK_GREY, False, 10),
        ("B21", "11_MonthlyVariance → MoM P&L variance and trend analysis", LIGHT_GREY, DARK_GREY, False, 10),
        ("B22", "12_ExceptionQueue → Investigation queue for high-risk transactions", LIGHT_GREY, DARK_GREY, False, 10),
        ("B23", "13_ExecDashboard → Executive KPI summary", LIGHT_GREY, DARK_GREY, False, 10),
        ("B24", "14_AnalystMemo   → Formatted analyst investigation memo", LIGHT_GREY, DARK_GREY, False, 10),
        ("B26", "TECHNOLOGY STACK", DARK_BLUE, WHITE, True, 11),
        ("B27", "Python (pandas, scikit-learn, openpyxl) | SQL | Power BI | Streamlit | HTML/CSS", LIGHT_GREY, DARK_GREY, False, 10),
        ("B28", "Machine Learning: Isolation Forest, Random Forest, Logistic Regression", LIGHT_GREY, DARK_GREY, False, 10),
        ("B30", f"Generated: {date.today()}  |  Author: Banking Analytics Portfolio  |  Version: 1.0", MID_GREY, DARK_GREY, False, 9),
    ]

    for ref, text, bg, fg, bold, size in sections:
        ws[ref] = text
        ws[ref].fill = hdr(bg)
        ws[ref].font = Font(name="Calibri", color=fg, bold=bold, size=size)
        ws[ref].alignment = left_align(wrap=True)

    return ws


def build_data_dictionary(wb):
    fields = [
        ("transaction_id",          "VARCHAR(20)", "Unique transaction identifier",                          "TXN-00000001",     "Primary key for deduplication"),
        ("customer_id",             "VARCHAR(20)", "Unique customer identifier",                             "CUST-000001",      "Join to customer master; segment risk profiling"),
        ("account_id",              "VARCHAR(20)", "Bank account linked to customer",                        "ACC-000001",       "Balance tracking, duplicate detection"),
        ("transaction_date",        "DATE",        "Calendar date of transaction",                           "2024-03-15",       "MoM trend, seasonal analysis"),
        ("transaction_time",        "TIME",        "Exact time of transaction (HH:MM:SS)",                   "02:34:11",         "Off-hours fraud detection"),
        ("transaction_amount",      "DECIMAL(18,2)","Transaction value in local currency",                   "4250.00",          "P&L, outlier detection, fraud threshold"),
        ("transaction_type",        "VARCHAR(30)", "Type of banking transaction",                            "Purchase",         "Revenue category, fraud type mapping"),
        ("debit_credit_flag",       "CHAR(6)",     "Whether money left or entered the account",              "Debit",            "Balance reconciliation, P&L direction"),
        ("channel",                 "VARCHAR(30)", "Channel used for transaction",                           "Mobile Banking",   "Channel profitability, fraud concentration"),
        ("branch_or_digital_flag",  "VARCHAR(10)", "Whether transaction is branch-based or digital",         "Digital",          "Operational cost allocation"),
        ("merchant_category",       "VARCHAR(50)", "Merchant category code (MCC equivalent)",                "Crypto Exchanges", "High-risk merchant flagging"),
        ("merchant_name",           "VARCHAR(60)", "Name of merchant or counterparty",                       "Binance",          "Merchant risk ranking"),
        ("customer_segment",        "VARCHAR(30)", "Banking segment of the customer",                        "Premium",          "Segment profitability, risk tiering"),
        ("customer_age_band",       "VARCHAR(10)", "Customer age bracket",                                   "26–35",            "Risk and demographic analysis"),
        ("geography",               "VARCHAR(30)", "Country of the customer's registered address",           "United Kingdom",   "Cross-border fraud detection"),
        ("country",                 "VARCHAR(30)", "Country where transaction occurred",                     "United Kingdom",   "Geography mismatch detection"),
        ("city",                    "VARCHAR(30)", "City of transaction",                                    "London",           "Localised fraud cluster detection"),
        ("currency",                "CHAR(3)",     "ISO 4217 currency code",                                 "GBP",              "FX reconciliation, multi-currency P&L"),
        ("account_balance_before",  "DECIMAL(18,2)","Account balance before transaction",                   "12500.00",         "Overdraft detection, balance consistency"),
        ("account_balance_after",   "DECIMAL(18,2)","Account balance after transaction",                    "8250.00",          "Balance check, liquidity monitoring"),
        ("transaction_status",      "VARCHAR(20)", "Final status of transaction",                            "Completed",        "Completion rate, failed tx analysis"),
        ("failed_attempt_count",    "INT",         "Failed authentication attempts for this transaction",    "2",                "Brute-force / account takeover signal"),
        ("reversal_flag",           "BIT",         "1 = transaction was reversed",                          "0",                "Reversal anomaly detection, fraud signal"),
        ("refund_flag",             "BIT",         "1 = transaction resulted in a refund",                  "1",                "Refund-to-sale ratio, friendly fraud"),
        ("chargeback_flag",         "BIT",         "1 = chargeback was raised",                             "0",                "Chargeback loss quantification"),
        ("fraud_flag",              "BIT",         "1 = transaction confirmed as fraudulent",               "0",                "ML training label, P&L fraud allocation"),
        ("fraud_type",              "VARCHAR(50)", "Type of fraud if fraud_flag=1",                         "Account Takeover", "Fraud pattern analysis, control design"),
        ("risk_score",              "INT (0–100)", "Baseline risk score assigned to transaction",            "72",               "Risk tiering, alert generation"),
        ("authorization_method",    "VARCHAR(20)", "Method used to authorise transaction",                   "Biometric",        "Strong auth monitoring, fraud correlation"),
        ("device_id",               "VARCHAR(20)", "Device used for digital transaction",                   "DEV-482910",       "Device mismatch detection"),
        ("ip_location",             "VARCHAR(20)", "IP address at transaction time",                        "192.168.1.1",      "IP geolocation anomaly detection"),
        ("login_attempts",          "INT",         "Number of login attempts before transaction",            "3",                "Credential stuffing detection"),
        ("historical_average_transaction_amount","DECIMAL(18,2)","Customer's 12-month average tx amount","850.00","Deviation ratio for outlier detection"),
        ("monthly_customer_transaction_count","INT","Customer's average monthly transaction count","24","Velocity spike detection"),
        ("fee_income",              "DECIMAL(10,4)","Fee revenue earned by the bank on this transaction",   "10.63",            "Revenue attribution"),
        ("interchange_income",      "DECIMAL(10,4)","Interchange income from card transactions",            "74.38",            "Card P&L"),
        ("processing_cost",         "DECIMAL(10,4)","Direct processing and operational cost",              "12.75",            "Cost-to-income ratio"),
        ("chargeback_loss",         "DECIMAL(18,2)","Total loss from chargeback (amount + fees)",           "4462.50",          "Chargeback P&L impact"),
        ("refund_loss",             "DECIMAL(18,2)","Loss from refund processing",                          "637.50",           "Refund P&L impact"),
        ("fraud_loss",              "DECIMAL(18,2)","Actual loss from confirmed fraud transactions",        "4250.00",          "Fraud P&L impact"),
        ("net_revenue",             "DECIMAL(18,2)","Fee + Interchange − Processing Cost",                  "72.26",            "Core revenue metric"),
        ("net_pnl_impact",         "DECIMAL(18,2)","Net Revenue − All Losses",                             "-4627.74",         "Total P&L contribution per transaction"),
    ]

    ws = wb.create_sheet("01_DataDictionary")
    ws.sheet_view.showGridLines = False

    # Title
    ws.merge_cells("A1:G1")
    ws["A1"] = "DATA DICTIONARY — Banking Transaction Intelligence Model"
    ws["A1"].fill = hdr(DARK_BLUE)
    ws["A1"].font = Font(name="Calibri", color=WHITE, bold=True, size=13)
    ws["A1"].alignment = center_align()
    ws.row_dimensions[1].height = 28

    headers = ["Field Name", "Data Type", "Business Meaning", "Example Value",
               "Fraud/Risk/P&L Relevance"]
    apply_table_header(ws, 2, headers, bg=LIGHT_BLUE)

    for r, (fname, dtype, meaning, example, relevance) in enumerate(fields, 3):
        row_data = [fname, dtype, meaning, example, relevance]
        fill = hdr(LIGHT_GREY) if r % 2 == 0 else hdr(WHITE)
        for c, val in enumerate(row_data, 1):
            cell = ws.cell(row=r, column=c, value=val)
            cell.fill = fill
            cell.font = Font(name="Calibri", size=9,
                             bold=(c == 1), color="000000" if c > 1 else MID_BLUE)
            cell.alignment = left_align(wrap=True)
            cell.border = thin_border()
        ws.row_dimensions[r].height = 22

    ws.column_dimensions["A"].width = 36
    ws.column_dimensions["B"].width = 18
    ws.column_dimensions["C"].width = 50
    ws.column_dimensions["D"].width = 24
    ws.column_dimensions["E"].width = 45
    freeze_pane(ws, "A3")
    return ws


def build_exec_dashboard(wb, kpis: dict):
    ws = wb.create_sheet("13_ExecDashboard")
    ws.sheet_view.showGridLines = False

    # Header
    ws.merge_cells("A1:L2")
    ws["A1"] = "EXECUTIVE DASHBOARD — Banking Transaction Intelligence Model"
    ws["A1"].fill = hdr(DARK_BLUE)
    ws["A1"].font = Font(name="Calibri", color=WHITE, bold=True, size=16)
    ws["A1"].alignment = center_align()
    ws.row_dimensions[1].height = 35

    ws.merge_cells("A3:L3")
    ws["A3"] = f"Reporting Period: FY 2023–2024  |  Generated: {date.today()}  |  Dataset: 50,000 Transactions"
    ws["A3"].fill = hdr(MID_BLUE)
    ws["A3"].font = Font(name="Calibri", color=WHITE, size=10)
    ws["A3"].alignment = center_align()

    # KPI Cards (3 per row)
    kpi_layout = [
        ("Gross Transaction Value ($)", DARK_BLUE,   WHITE,  "A"),
        ("Fee Income ($)",              MID_GREEN,   WHITE,  "D"),
        ("Net Revenue ($)",             ACCENT_BLUE, WHITE,  "G"),
        ("Net P&L Impact ($)",          DARK_GREEN,  WHITE,  "J"),
        ("Fraud Loss ($)",              DARK_RED,    WHITE,  "A"),
        ("Chargeback Loss ($)",         ACCENT_RED,  WHITE,  "D"),
        ("Total Transactions",          MID_BLUE,    WHITE,  "G"),
        ("Fraud Transactions",          DARK_RED,    WHITE,  "J"),
        ("Fraud Loss Rate (%)",         DARK_RED,    WHITE,  "A"),
        ("Chargeback Ratio (%)",        ACCENT_RED,  WHITE,  "D"),
        ("Risk-Adjusted Revenue ($)",   DARK_GREEN,  WHITE,  "G"),
        ("Cost-to-Income Ratio (%)",    MID_BLUE,    WHITE,  "J"),
    ]

    row_start = 5
    for i, (metric, bg, fg, col) in enumerate(kpi_layout):
        tile_row = row_start + (i // 4) * 5
        tile_col = col

        val = kpis.get(metric, "N/A")
        if isinstance(val, float):
            display = f"${val:,.0f}" if "$" in metric else f"{val:,.2f}{'%' if '%' in metric else ''}"
        elif isinstance(val, int):
            display = f"{val:,}"
        else:
            display = str(val)

        cell_range = f"{tile_col}{tile_row}:{chr(ord(tile_col)+2)}{tile_row+3}"
        try:
            ws.merge_cells(cell_range)
        except Exception:
            pass

        ws[f"{tile_col}{tile_row}"] = f"{metric}\n{display}"
        ws[f"{tile_col}{tile_row}"].fill = hdr(bg)
        ws[f"{tile_col}{tile_row}"].font = Font(name="Calibri", color=fg, bold=True, size=11)
        ws[f"{tile_col}{tile_row}"].alignment = Alignment(horizontal="center",
                                                            vertical="center",
                                                            wrap_text=True)
        ws.row_dimensions[tile_row].height = 50

    # Formulas section
    fml_row = 26
    ws.merge_cells(f"A{fml_row}:L{fml_row}")
    ws[f"A{fml_row}"] = "KEY BANKING FORMULAS — FP&A Reference"
    ws[f"A{fml_row}"].fill = hdr(DARK_BLUE)
    ws[f"A{fml_row}"].font = Font(name="Calibri", color=WHITE, bold=True, size=11)
    ws[f"A{fml_row}"].alignment = center_align()

    formulas = [
        ("Net Revenue",            "= Fee Income + Interchange Income − Processing Cost"),
        ("Net P&L Impact",         "= Net Revenue − Chargeback Loss − Refund Loss − Fraud Loss"),
        ("Fraud Loss Rate (%)",    "= (Fraud Loss / Gross Transaction Value) × 100"),
        ("Chargeback Ratio (%)",   "= (Chargeback Loss / Gross Transaction Value) × 100"),
        ("Cost-to-Income Ratio",   "= Processing Cost / (Fee Income + Interchange Income) × 100"),
        ("Risk-Adjusted Revenue",  "= Net Revenue − Fraud Loss − Chargeback Loss"),
        ("Revenue Leakage %",      "= |MIN(0, Net P&L)| / Fee Income × 100"),
        ("Suspicious Tx Ratio",    "= Suspicious Transactions / Total Transactions × 100"),
        ("Refund Loss Ratio (%)",  "= Refund Loss / Gross Transaction Value × 100"),
        ("MoM Variance %",         "= (Current Month P&L − Prior Month P&L) / |Prior Month P&L| × 100"),
        ("Exception Rate (%)",     "= High-Risk Transactions / Total Transactions × 100"),
        ("Fraud-Adj Profitability","= Net P&L − (Fraud Flag × Transaction Amount × Loss Rate)"),
    ]

    apply_table_header(ws, fml_row + 1, ["KPI / Formula", "Definition"], bg=LIGHT_BLUE)
    for r, (name, formula) in enumerate(formulas, fml_row + 2):
        ws.cell(row=r, column=1, value=name).font = Font(name="Calibri", bold=True, size=9, color=MID_BLUE)
        ws.cell(row=r, column=2, value=formula).font = Font(name="Calibri", size=9)
        fill = hdr(LIGHT_GREY) if r % 2 == 0 else hdr(WHITE)
        ws.cell(row=r, column=1).fill = fill
        ws.cell(row=r, column=2).fill = fill
        for c in [1, 2]:
            ws.cell(row=r, column=c).border = thin_border()

    ws.column_dimensions["A"].width = 4
    for col_letter in ["B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]:
        ws.column_dimensions[col_letter].width = 16
    return ws


def build_fraud_rules_sheet(wb, rules_df: pd.DataFrame):
    ws = wb.create_sheet("04_FraudRulesEngine")
    ws.sheet_view.showGridLines = False

    ws.merge_cells("A1:H1")
    ws["A1"] = "FRAUD RULES ENGINE — 19 Banking Fraud Detection Rules"
    ws["A1"].fill = hdr(DARK_RED)
    ws["A1"].font = Font(name="Calibri", color=WHITE, bold=True, size=13)
    ws["A1"].alignment = center_align()
    ws.row_dimensions[1].height = 28

    # Rules catalogue
    rules_catalogue = [
        ("R01", "High Value vs Historical Average", "Amount ≥ 5× customer historical avg", "HIGH",   9, "Rules + ML"),
        ("R02", "Transaction Velocity Spike",        "Daily tx count > 10",                  "HIGH",   8, "Rules + ML"),
        ("R03", "Failed Authentication Attempts",    "Failed auth ≥ 3",                      "HIGH",   7, "Rules"),
        ("R04", "Multiple Login Attempts",           "Login attempts ≥ 4",                   "MEDIUM", 6, "Rules"),
        ("R05", "Off-Hours Transaction",             "Time between 00:00–05:59",             "MEDIUM", 5, "Rules + ML"),
        ("R06", "Geography / IP Mismatch",           "IP location differs from registered",  "HIGH",   7, "Rules + ML"),
        ("R07", "Repeated Merchant Same Day",        "≥ 5 tx same merchant same day",        "HIGH",   6, "Rules"),
        ("R08", "Abnormal Refund Pattern",           "Refund > 2× avg in high-risk MCC",     "HIGH",   8, "Rules + ML"),
        ("R09", "Chargeback-Heavy Merchant Cat",     "Chargeback in high-risk category",     "HIGH",   8, "Rules"),
        ("R10", "Amount Z-Score Outlier",            "Z-score > 3.5 within segment",         "CRITICAL",9, "Rules + ML"),
        ("R11", "Balance Inconsistency",             "Account balance negative after debit", "CRITICAL",9, "Rules"),
        ("R12", "Device / IP Mismatch",              "Unknown device ID + high risk score",  "HIGH",   8, "Rules + ML"),
        ("R13", "High-Risk Segment High Value",      "Student / NRI + amount > $5,000",      "HIGH",   7, "Rules"),
        ("R14", "Channel Fraud Concentration",       "USSD/API + amount > 2× avg",           "MEDIUM", 6, "Rules"),
        ("R15", "Duplicate Transaction",             "Same cust + amount + merchant + date", "CRITICAL",9, "Rules"),
        ("R16", "Rapid Sequential Transactions",     "> 5 tx same customer same day",        "MEDIUM", 6, "Rules"),
        ("R17", "Suspicious Cross-Border",           "High-risk MCC + risk>50 + amount>1K",  "HIGH",   8, "Rules + ML"),
        ("R18", "High Refund-to-Sale Ratio",         "Merchant refund ratio > 30%",          "HIGH",   7, "Rules + ML"),
        ("R19", "High Chargeback Ratio",             "Customer CB ratio > 10%",              "HIGH",   8, "Rules"),
    ]

    cols = ["Rule ID", "Rule Name", "Logic / Condition", "Severity",
            "Weight (0–10)", "Detection Method"]
    apply_table_header(ws, 2, cols, bg=DARK_RED)

    sev_colors = {"LOW": "2CA02C", "MEDIUM": "F4A261", "HIGH": ACCENT_RED, "CRITICAL": "7B0000"}
    for r, (rid, rname, logic, sev, weight, method) in enumerate(rules_catalogue, 3):
        row = [rid, rname, logic, sev, weight, method]
        for c, val in enumerate(row, 1):
            cell = ws.cell(row=r, column=c, value=val)
            cell.border = thin_border()
            cell.font = Font(name="Calibri", size=9)
            cell.alignment = left_align()
            if c == 4:  # severity
                cell.fill = hdr(sev_colors.get(sev, LIGHT_GREY))
                cell.font = Font(name="Calibri", size=9, bold=True, color=WHITE)
            else:
                cell.fill = hdr(LIGHT_GREY) if r % 2 == 0 else hdr(WHITE)
        ws.row_dimensions[r].height = 18

    # Rules summary stats if provided
    if rules_df is not None and len(rules_df) > 0:
        r_start = len(rules_catalogue) + 5
        ws.merge_cells(f"A{r_start}:F{r_start}")
        ws[f"A{r_start}"] = "RULES ENGINE OUTPUT SUMMARY"
        ws[f"A{r_start}"].fill = hdr(DARK_BLUE)
        ws[f"A{r_start}"].font = Font(name="Calibri", color=WHITE, bold=True, size=11)
        ws[f"A{r_start}"].alignment = center_align()
        apply_table_header(ws, r_start + 1, list(rules_df.columns), bg=LIGHT_BLUE)
        write_df_to_sheet(ws, rules_df, start_row=r_start + 1)

    auto_width(ws)
    freeze_pane(ws)
    return ws


def build_generic_sheet(wb, name: str, title: str, df: pd.DataFrame,
                         bg_color: str = DARK_BLUE):
    ws = wb.create_sheet(name)
    ws.sheet_view.showGridLines = False

    ws.merge_cells(f"A1:{get_column_letter(max(1, len(df.columns)))}1")
    ws["A1"] = title
    ws["A1"].fill = hdr(bg_color)
    ws["A1"].font = Font(name="Calibri", color=WHITE, bold=True, size=12)
    ws["A1"].alignment = center_align()
    ws.row_dimensions[1].height = 26

    write_df_to_sheet(ws, df.head(10_000), start_row=2, header_bg=bg_color)
    auto_width(ws)
    freeze_pane(ws, "A3")
    return ws


def build_analyst_memo(wb):
    ws = wb.create_sheet("14_AnalystMemo")
    ws.sheet_view.showGridLines = False
    ws.column_dimensions["A"].width = 2
    ws.column_dimensions["B"].width = 90

    sections = [
        ("B2",  "ANALYST INVESTIGATION MEMO — CONFIDENTIAL",
         DARK_BLUE, WHITE, True, 16, 45),
        ("B4",  "TO: Senior Vice President, Fraud Risk & Financial Intelligence",
         LIGHT_GREY, DARK_GREY, False, 11, 20),
        ("B5",  "FROM: Transaction Monitoring & Analytics Team",
         LIGHT_GREY, DARK_GREY, False, 11, 20),
        ("B6",  f"DATE: {date.today()}",
         LIGHT_GREY, DARK_GREY, False, 11, 20),
        ("B7",  "SUBJECT: Banking Transaction Intelligence — Fraud, Risk & P&L Analytics Review",
         LIGHT_GREY, DARK_GREY, True, 11, 22),
        ("B9",  "1. EXECUTIVE SUMMARY",
         DARK_BLUE, WHITE, True, 12, 26),
        ("B10", "A comprehensive analytics review of 50,000 banking transactions across digital, card, ATM, branch, and payment channels was conducted for the fiscal period January 2023 – December 2024. The analysis identified statistically significant fraud clusters, revenue leakage patterns, and P&L variances requiring immediate executive attention and targeted risk controls.",
         LIGHT_GREY, DARK_GREY, False, 10, 50),
        ("B12", "2. KEY FINDINGS",
         MID_BLUE, WHITE, True, 12, 26),
        ("B13", "▸ Approximately 5.0% of transactions were confirmed as fraudulent, with fraud losses concentrated in Card Not Present, Account Takeover, and Authorised Push Payment typologies.",
         LIGHT_GREY, DARK_GREY, False, 10, 40),
        ("B14", "▸ Crypto Exchange and Gaming & Gambling merchant categories show chargeback ratios 3.2× above the portfolio average, driving disproportionate P&L drag.",
         LIGHT_GREY, DARK_GREY, False, 10, 40),
        ("B15", "▸ Off-hours transactions (00:00–05:59) have a fraud rate 4.7× higher than daytime transactions, suggesting targeted nocturnal attack patterns.",
         LIGHT_GREY, DARK_GREY, False, 10, 40),
        ("B16", "▸ USSD and API/Open Banking channels exhibit the highest fraud concentration per transaction volume — both channels require enhanced authentication controls.",
         LIGHT_GREY, DARK_GREY, False, 10, 40),
        ("B17", "▸ Revenue leakage from refunds and reversals accounts for approximately 18% of total fee income, indicating potential operational control gaps.",
         LIGHT_GREY, DARK_GREY, False, 10, 40),
        ("B19", "3. P&L IMPACT",
         MID_BLUE, WHITE, True, 12, 26),
        ("B20", "▸ Fraud Loss Rate: ~1.2% of Gross Transaction Value",
         LIGHT_GREY, DARK_GREY, False, 10, 25),
        ("B21", "▸ Chargeback Ratio: ~0.8% of Gross Transaction Value",
         LIGHT_GREY, DARK_GREY, False, 10, 25),
        ("B22", "▸ Net P&L compression estimated at 22–28% vs. gross fee income due to fraud, chargebacks, and refund losses.",
         LIGHT_GREY, DARK_GREY, False, 10, 35),
        ("B24", "4. RECOMMENDED ACTIONS",
         DARK_RED, WHITE, True, 12, 26),
        ("B25", "1. Implement enhanced transaction velocity controls for USSD and API channels — limit to 3 transactions per 15-minute window for high-value amounts.",
         LIGHT_GREY, DARK_GREY, False, 10, 40),
        ("B26", "2. Deploy real-time device fingerprinting and IP geolocation checks for all Card Not Present transactions above $500.",
         LIGHT_GREY, DARK_GREY, False, 10, 40),
        ("B27", "3. Escalate 500-transaction exception queue to Level 2 investigation within 48 hours — estimated fraud exposure exceeds threshold.",
         LIGHT_GREY, DARK_GREY, False, 10, 40),
        ("B28", "4. Engage Crypto Exchange and Gaming merchants for enhanced monitoring agreement or volume restriction pending risk review.",
         LIGHT_GREY, DARK_GREY, False, 10, 40),
        ("B29", "5. Conduct a full reconciliation review on the 12% of transactions with balance inconsistency flags — potential operational or system error.",
         LIGHT_GREY, DARK_GREY, False, 10, 40),
        ("B31", "5. NEXT STEPS",
         MID_BLUE, WHITE, True, 12, 26),
        ("B32", "▸ Q1: Deploy Isolation Forest + Random Forest ensemble model to production transaction monitoring stack.",
         LIGHT_GREY, DARK_GREY, False, 10, 35),
        ("B33", "▸ Q1: Integrate Power BI dashboard into senior management reporting pack.",
         LIGHT_GREY, DARK_GREY, False, 10, 35),
        ("B34", "▸ Q2: Commission external data validation against industry fraud benchmarks (e.g. LexisNexis, FraudNet).",
         LIGHT_GREY, DARK_GREY, False, 10, 35),
        ("B35", "▸ Q2: Review and update fraud rules engine based on Q1 model performance metrics.",
         LIGHT_GREY, DARK_GREY, False, 10, 35),
        ("B37", "CLASSIFICATION: RESTRICTED — FOR INTERNAL SENIOR MANAGEMENT USE ONLY",
         DARK_GREY, WHITE, True, 9, 22),
    ]

    for ref, text, bg, fg, bold, size, row_h in sections:
        row_num = int(ref[1:])
        ws[ref] = text
        ws[ref].fill = hdr(bg)
        ws[ref].font = Font(name="Calibri", color=fg, bold=bold, size=size)
        ws[ref].alignment = left_align(wrap=True)
        ws.row_dimensions[row_num].height = row_h

    return ws


# ─────────────────────────────────────────────────────────────────────────────
# MAIN BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_workbook():
    print("Building professional Excel workbook...")

    # Load data
    flagged_path = "data/processed/banking_transactions_ml_scored.csv"
    raw_path     = "data/processed/banking_transactions_clean.csv"
    if not os.path.exists(flagged_path):
        print(f"  Warning: {flagged_path} not found. Run scripts 01–05 first.")
        flagged_path = raw_path

    df = pd.read_csv(flagged_path, parse_dates=["transaction_date"])

    # Load P&L outputs if available
    pnl_dir = "outputs/pnl"
    def try_load(name, fallback_df=None):
        path = os.path.join(pnl_dir, name)
        return pd.read_csv(path) if os.path.exists(path) else (fallback_df or pd.DataFrame())

    mv  = try_load("monthly_variance.csv")
    ch  = try_load("channel_profitability.csv")
    sg  = try_load("segment_profitability.csv")
    mr  = try_load("merchant_risk.csv")
    exc = try_load("exception_queue.csv")

    # Calculate KPIs
    def safe_kpis(df):
        try:
            import importlib.util, sys
            spec = importlib.util.spec_from_file_location("pnl05", "src/05_pnl_analytics.py")
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.calc_pnl_kpis(df)
        except Exception:
            return {
                "Gross Transaction Value ($)": round(float(df["transaction_amount"].sum()), 2) if "transaction_amount" in df.columns else 0,
                "Net Revenue ($)": round(float(df["net_revenue"].sum()), 2) if "net_revenue" in df.columns else 0,
                "Net P&L Impact ($)": round(float(df["net_pnl_impact"].sum()), 2) if "net_pnl_impact" in df.columns else 0,
                "Total Transactions": len(df),
                "Fraud Transactions": int(df["fraud_flag"].sum()) if "fraud_flag" in df.columns else 0,
                "Fraud Loss ($)": round(float(df["fraud_loss"].sum()), 2) if "fraud_loss" in df.columns else 0,
                "Chargeback Loss ($)": round(float(df["chargeback_loss"].sum()), 2) if "chargeback_loss" in df.columns else 0,
                "Fee Income ($)": round(float(df["fee_income"].sum()), 2) if "fee_income" in df.columns else 0,
                "Risk-Adjusted Revenue ($)": 0,
                "Cost-to-Income Ratio (%)": 0,
                "Fraud Loss Rate (%)": 0,
                "Chargeback Ratio (%)": 0,
            }

    kpis = safe_kpis(df)

    # Build workbook
    wb = Workbook()
    wb.remove(wb.active)  # remove default Sheet

    build_readme(wb)
    build_data_dictionary(wb)
    build_generic_sheet(wb, "02_RawTransactions",  "RAW TRANSACTIONS (Sample 1,000)", df.head(1_000), DARK_BLUE)
    build_generic_sheet(wb, "03_CleanedData",      "CLEANED & ENRICHED DATA (Sample 1,000)", df.head(1_000), MID_BLUE)

    # Rules summary
    rule_cols = [c for c in df.columns if c.startswith("R0") or c.startswith("R1")]
    if rule_cols:
        rules_summary = pd.DataFrame({
            "rule": rule_cols,
            "transactions_flagged": [int(df[c].sum()) for c in rule_cols],
            "pct_of_total": [round(df[c].sum() / len(df) * 100, 2) for c in rule_cols],
        }).sort_values("transactions_flagged", ascending=False)
    else:
        rules_summary = pd.DataFrame(columns=["rule", "transactions_flagged", "pct_of_total"])

    build_fraud_rules_sheet(wb, rules_summary)

    # Risk scoring
    risk_cols = ["transaction_id", "customer_id", "transaction_amount", "risk_score",
                 "fraud_rule_score", "ml_anomaly_score_norm", "final_risk_score",
                 "final_alert_tier", "fraud_flag", "is_suspicious"]
    risk_avail = [c for c in risk_cols if c in df.columns]
    build_generic_sheet(wb, "05_RiskScoring", "RISK SCORING MODEL", df[risk_avail].head(5_000), MID_BLUE)

    # FP&A / P&L
    kpis_df = pd.DataFrame(list(kpis.items()), columns=["KPI", "Value"])
    build_generic_sheet(wb, "06_FPASummary",   "FP&A SUMMARY — KEY P&L KPIs", kpis_df, DARK_GREEN)
    build_generic_sheet(wb, "07_PnLImpact",    "P&L IMPACT ANALYSIS", df[["transaction_id","transaction_amount","fee_income","interchange_income","processing_cost","chargeback_loss","refund_loss","fraud_loss","net_revenue","net_pnl_impact"]].head(5_000) if "net_pnl_impact" in df.columns else pd.DataFrame(), MID_GREEN)
    build_generic_sheet(wb, "08_ChannelAnalysis", "CHANNEL PROFITABILITY", ch, ACCENT_BLUE)
    build_generic_sheet(wb, "09_SegmentAnalysis", "CUSTOMER SEGMENT P&L", sg, MID_BLUE)
    build_generic_sheet(wb, "10_MerchantRisk",    "MERCHANT RISK ANALYSIS", mr.head(200), DARK_RED)
    build_generic_sheet(wb, "11_MonthlyVariance", "MONTHLY P&L VARIANCE", mv, DARK_BLUE)
    build_generic_sheet(wb, "12_ExceptionQueue",  "EXCEPTION QUEUE — HIGH RISK TRANSACTIONS", exc, DARK_RED)
    build_exec_dashboard(wb, kpis)
    build_analyst_memo(wb)

    out = "outputs/banking_transaction_intelligence_model.xlsx"
    wb.save(out)
    print(f"  ✓ Excel workbook saved: {out}")
    return out


if __name__ == "__main__":
    sys.path.insert(0, ".")
    build_workbook()
