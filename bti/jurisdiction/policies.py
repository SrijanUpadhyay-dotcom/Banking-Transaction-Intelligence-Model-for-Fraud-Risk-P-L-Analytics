"""
Jurisdiction policies for the eight markets in the BTI book.

Each policy drives decisioning (loss allocation, whether a decline must offer
a human-review route, whether step-up authentication is expected) and records
the regulatory map a compliance team reviews before go-live.

This is a starting map for compliance review, not legal advice. Items marked
"confirm with local counsel" are ones where requirements are sector-specific
or have changed recently.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

DISCLAIMER = ("Regulatory references are a starting map for compliance review, not legal advice. "
              "Confirm with local counsel before production use.")

DEFAULT_LOSS_GIVEN_FRAUD = 0.75


@dataclass(frozen=True)
class JurisdictionPolicy:
    country: str
    iso2: str
    currency: str
    supervisors: List[str]
    model_governance: List[str]
    customer_liability: str
    automated_decisions: str
    strong_authentication: str
    suspicious_reporting: str
    tipping_off: str
    data_protection: str
    data_residency: str
    loss_given_fraud: float = DEFAULT_LOSS_GIVEN_FRAUD
    decline_requires_human_review_route: bool = False
    psd2_tra: bool = False
    notes: List[str] = field(default_factory=list)


POLICIES: Dict[str, JurisdictionPolicy] = {p.iso2: p for p in [
    JurisdictionPolicy(
        country="United States", iso2="US", currency="USD",
        supervisors=["Federal Reserve", "OCC", "FDIC", "CFPB"],
        model_governance=["SR 11-7 / OCC Bulletin 2011-12 (model risk management)"],
        customer_liability="Regulation E caps consumer liability for unauthorised electronic fund transfers "
                           "reported on time, so most debit/EFT fraud loss sits with the bank.",
        automated_decisions="No general statutory right to human review. ECOA/Reg B adverse-action reasons apply "
                            "to credit decisions; point-of-sale authorisation refusals are generally excluded "
                            "(12 CFR 1002.2(c)(2)(iii)).",
        strong_authentication="No statutory SCA; FFIEC authentication guidance expects risk-based controls.",
        suspicious_reporting="SAR to FinCEN within 30 calendar days of initial detection (Bank Secrecy Act).",
        tipping_off="31 U.S.C. 5318(g)(2) prohibits disclosing that a SAR was filed.",
        data_protection="Gramm-Leach-Bliley Act; state privacy laws.",
        data_residency="No general federal data-localisation requirement.",
        loss_given_fraud=0.85,
    ),
    JurisdictionPolicy(
        country="United Kingdom", iso2="GB", currency="GBP",
        supervisors=["PRA", "FCA", "Payment Systems Regulator"],
        model_governance=["PRA SS1/23 model risk management principles (in force 17 May 2024)",
                          "FCA Consumer Duty"],
        customer_liability="PSR mandatory reimbursement for authorised push payment scams from 7 Oct 2024 "
                           "(cap £85,000, shared between sending and receiving firms); PSRs 2017 refund rules "
                           "for unauthorised payments.",
        automated_decisions="UK GDPR Art. 22: customers can obtain human intervention and contest solely "
                            "automated decisions with significant effects.",
        strong_authentication="UK SCA-RTS; transaction-risk-analysis exemption follows the EU reference "
                              "fraud-rate structure (confirm current thresholds with the FCA RTS).",
        suspicious_reporting="SAR to the UK Financial Intelligence Unit (NCA) under POCA 2002.",
        tipping_off="POCA 2002 s.333A.",
        data_protection="UK GDPR and Data Protection Act 2018.",
        data_residency="No general localisation requirement; international transfer rules apply.",
        loss_given_fraud=0.90,
        decline_requires_human_review_route=True,
        psd2_tra=True,
    ),
    JurisdictionPolicy(
        country="Germany", iso2="DE", currency="EUR",
        supervisors=["BaFin", "Deutsche Bundesbank", "ECB (SSM, significant institutions)"],
        model_governance=["BaFin MaRisk", "EBA Guidelines on ICT and security risk management",
                          "EU AI Act: Annex III 5(b) excludes AI used to detect financial fraud from the "
                          "high-risk creditworthiness category"],
        customer_liability="PSD2: payer refunded for unauthorised payments; liability capped at EUR 50 absent "
                           "fraud or gross negligence.",
        automated_decisions="GDPR Art. 22: right to human intervention, to express a view and to contest.",
        strong_authentication="PSD2 SCA (RTS 2018/389); TRA exemption available below reference fraud rates.",
        suspicious_reporting="Suspicious activity report to FIU Germany (Zentralstelle für "
                             "Finanztransaktionsuntersuchungen) under GwG §43.",
        tipping_off="GwG §47.",
        data_protection="GDPR and BDSG.",
        data_residency="No general localisation requirement; GDPR transfer rules apply.",
        loss_given_fraud=0.85,
        decline_requires_human_review_route=True,
        psd2_tra=True,
    ),
    JurisdictionPolicy(
        country="India", iso2="IN", currency="INR",
        supervisors=["Reserve Bank of India"],
        model_governance=["RBI Master Directions on Fraud Risk Management (July 2024)"],
        customer_liability="RBI customer-liability framework (2017): zero customer liability for third-party "
                           "breaches reported within three working days.",
        automated_decisions="DPDP Act 2023 obligations on processing; no GDPR-style Art. 22 right.",
        strong_authentication="Additional factor of authentication required for card-not-present transactions "
                              "(small-value exemptions apply).",
        suspicious_reporting="STR to FIU-IND under PMLA within seven working days of concluding suspicion.",
        tipping_off="PMLA framework prohibits tipping off the customer.",
        data_protection="Digital Personal Data Protection Act 2023.",
        data_residency="RBI directive on Storage of Payment System Data (April 2018): payment system data "
                       "must be stored only in India — deploy scoring and logs in-country.",
        loss_given_fraud=0.85,
        notes=["Scoring infrastructure for Indian payment data must run in India."],
    ),
    JurisdictionPolicy(
        country="Singapore", iso2="SG", currency="SGD",
        supervisors=["Monetary Authority of Singapore"],
        model_governance=["MAS FEAT principles", "MAS Technology Risk Management Guidelines"],
        customer_liability="E-Payments User Protection Guidelines; Shared Responsibility Framework for "
                           "phishing scams (effective 16 Dec 2024).",
        automated_decisions="PDPA obligations; FEAT transparency expectations.",
        strong_authentication="MAS TRM guidelines expect multi-factor authentication for high-risk transactions.",
        suspicious_reporting="STR to the Suspicious Transaction Reporting Office (STRO).",
        tipping_off="CDSA prohibits tipping off.",
        data_protection="Personal Data Protection Act 2012.",
        data_residency="No general localisation requirement; MAS outsourcing guidelines apply.",
    ),
    JurisdictionPolicy(
        country="Hong Kong", iso2="HK", currency="HKD",
        supervisors=["Hong Kong Monetary Authority"],
        model_governance=["HKMA high-level principles on artificial intelligence (2019)",
                          "HKMA consumer-protection principles for big data analytics and AI (2019)"],
        customer_liability="Code of Banking Practice limits customer liability for unauthorised transactions "
                           "absent fraud or gross negligence.",
        automated_decisions="PDPO obligations; HKMA expects explainability and a route to human review.",
        strong_authentication="HKMA expects two-factor authentication for high-risk online transactions.",
        suspicious_reporting="STR to the Joint Financial Intelligence Unit (JFIU).",
        tipping_off="OSCO / DTROP prohibit tipping off.",
        data_protection="Personal Data (Privacy) Ordinance (Cap. 486).",
        data_residency="No general localisation requirement.",
    ),
    JurisdictionPolicy(
        country="United Arab Emirates", iso2="AE", currency="AED",
        supervisors=["Central Bank of the UAE"],
        model_governance=["CBUAE Model Management Standards and Guidance"],
        customer_liability="CBUAE consumer-protection regulation; confirm liability allocation with local counsel.",
        automated_decisions="Federal PDPL obligations; confirm with local counsel.",
        strong_authentication="CBUAE expects strong authentication for digital payments; confirm current rules.",
        suspicious_reporting="STR to the UAE Financial Intelligence Unit via goAML.",
        tipping_off="Federal AML law prohibits tipping off.",
        data_protection="Federal Decree-Law 45/2021 (PDPL).",
        data_residency="Sector-specific localisation may apply — confirm with local counsel.",
    ),
    JurisdictionPolicy(
        country="Nigeria", iso2="NG", currency="NGN",
        supervisors=["Central Bank of Nigeria"],
        model_governance=["Confirm model-governance expectations with the CBN and local counsel"],
        customer_liability="Confirm liability allocation for unauthorised transactions with local counsel.",
        automated_decisions="NDPA 2023 includes rights relating to automated decision-making; confirm scope.",
        strong_authentication="CBN requires two-factor authentication for electronic payments; confirm current rules.",
        suspicious_reporting="STR to the Nigerian Financial Intelligence Unit (NFIU).",
        tipping_off="Money Laundering (Prevention and Prohibition) Act prohibits tipping off.",
        data_protection="Nigeria Data Protection Act 2023.",
        data_residency="Localisation requirements may apply — confirm with local counsel.",
    ),
]}

_COUNTRY_ALIASES = {p.country.lower(): p.iso2 for p in POLICIES.values()}
_COUNTRY_ALIASES.update({"uae": "AE", "uk": "GB", "united kingdom": "GB", "usa": "US", "us": "US",
                         "great britain": "GB"})


def policy_for(country_or_iso: Optional[str]) -> Optional[JurisdictionPolicy]:
    if not country_or_iso:
        return None
    key = str(country_or_iso).strip()
    if key.upper() in POLICIES:
        return POLICIES[key.upper()]
    iso = _COUNTRY_ALIASES.get(key.lower())
    return POLICIES.get(iso) if iso else None


def policy_dict(policy: JurisdictionPolicy) -> Dict:
    return {**asdict(policy), "disclaimer": DISCLAIMER}


# ── PSD2 transaction-risk-analysis exemption (RTS 2018/389, Annex) ────────────
# Reference fraud rates by exemption threshold value (EUR).
TRA_REFERENCE_FRAUD_RATES = {
    "remote_card": [(500, 0.0001), (250, 0.0006), (100, 0.0013)],
    "credit_transfer": [(500, 0.00005), (250, 0.0001), (100, 0.00015)],
}


def tra_eligibility(fraud_value_eur: float, total_value_eur: float, payment_type: str = "remote_card") -> Dict:
    """
    The highest TRA exemption threshold a PSP may use, given its fraud rate
    (fraud value ÷ total value of remote transactions of that type, rolling 90 days).
    """
    if payment_type not in TRA_REFERENCE_FRAUD_RATES:
        raise ValueError(f"payment_type must be one of {sorted(TRA_REFERENCE_FRAUD_RATES)}")
    if total_value_eur <= 0:
        raise ValueError("total_value_eur must be positive")
    rate = fraud_value_eur / total_value_eur
    bands = [{"exemption_threshold_eur": t, "reference_fraud_rate": r, "eligible": rate <= r}
             for t, r in TRA_REFERENCE_FRAUD_RATES[payment_type]]
    eligible = [b["exemption_threshold_eur"] for b in bands if b["eligible"]]
    return {
        "payment_type": payment_type,
        "fraud_rate": round(rate, 7),
        "fraud_rate_bps": round(rate * 10_000, 3),
        "max_exemption_threshold_eur": max(eligible) if eligible else 0,
        "bands": bands,
        "note": "Rates must be computed per payment type over a rolling 90-day window and reported to the "
                "competent authority.",
    }
