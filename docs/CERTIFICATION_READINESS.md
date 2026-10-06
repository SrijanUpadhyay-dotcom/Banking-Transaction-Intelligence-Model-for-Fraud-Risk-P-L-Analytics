# BTI — Certification Readiness (SOC 2, ISO 27001, PCI DSS, Penetration Testing)

**BTI is not certified.** No code can award a certification:
- **SOC 2** is an attestation report by an independent CPA firm.
- **ISO/IEC 27001** is certified by an accredited certification body.
- **PCI DSS** compliance is validated by a Qualified Security Assessor (QSA)
  or a self-assessment, as the card brands require.
- **Penetration testing** must be done by an independent tester.

Each also covers the *organisation* that runs BTI (policies, people, suppliers,
facilities), not just the software.

Phase 10 builds what the software must contribute: technical controls,
automated self-testing, continuous evidence, scope reduction, and the material
an auditor and a tester need. This document maps it to each framework and
lists what the operating organisation still has to do.

Framework references are a starting map for the bank's or vendor's compliance
team and auditors, not legal or audit advice.

---

## 1. Who awards what, and how long it takes

| | Awarded by | What it covers | Typical path |
|---|---|---|---|
| **SOC 2 Type II** | Independent CPA firm (AICPA attestation) | Design *and operating effectiveness* of controls over an observation period, against the Trust Services Criteria (security, plus any of availability, confidentiality, processing integrity, privacy) | Readiness assessment → Type I (design, point in time) → 3–12-month observation → Type II report; then yearly |
| **ISO/IEC 27001:2022** | Accredited certification body (accredited by e.g. UKAS, ANAB, NABCB) | An information security management system (clauses 4–10) and the Annex A controls declared in the Statement of Applicability | ISMS build (scope, risk assessment, SoA, policies, internal audit, management review) → Stage 1 audit (documentation) → Stage 2 audit (implementation) → certificate for 3 years with annual surveillance audits |
| **PCI DSS v4.0.1** | QSA (Report on Compliance), or self-assessment questionnaire where permitted | Protection of cardholder data and the systems that touch it | Scope (data flows, cardholder data environment (CDE)) → gap assessment → remediation → ROC / SAQ yearly. A service provider with more than 300,000 card transactions a year per brand is typically Level 1: ROC by a QSA |
| **Penetration test** | Qualified, organisationally independent tester | Exploitable weaknesses in the application and its infrastructure | Scope and rules of engagement → test → report → fix → retest. At least yearly and after significant change (PCI DSS 11.4) |

**In India,** banks also assess technology vendors under RBI's IT outsourcing
directions (Master Direction on Outsourcing of IT Services, 2023). The
CERT-In directions apply as well: 6-hour incident reporting and 180-day log
retention in India. These are covered in docs/HA_DR_AND_RESIDENCY.md.

---

## 2. What Phase 10 built

| Control | Implementation | Evidence |
|---|---|---|
| **Authentication** | Every person and system is a *principal* with its own key, stored only as SHA-256; deactivation is immediate. The shared legacy key is off by default, and production refuses to start with the default key | `bti.security.principals`; `principals_review.json` |
| **Authorisation (least privilege)** | One role-based access table covers all 144 API routes. Default deny: a route with no rule is admin-only. Customer-level data is for analysts and auditors only; aggregate reports are for staff roles; auditors cannot write | `bti.security.access`; `access_matrix.json`; test: every route has a rule |
| **Segregation of duties / four-eyes** | The fields naming who acts (approver, author, validator, reviewer, analyst, checker) must equal the authenticated caller. So the registry's four-eyes rule (approver ≠ developer), rule approval (approver ≠ author) and validation sign-off cannot be faked by typing a name. A high-value case clearance needs the checker's own authenticated confirmation | API tests; audit log `CASE_CHECKED`, promotions, sign-offs |
| **Tamper-evident logging** | Hash-chained audit log with database triggers against update and delete, a daily sealed archive, and 7-year retention (Phase 3). Evidence packs are anchored in it | `audit_chain.json` |
| **Change management** | Immutable model registry with validation gates and four-eyes promotion; versioned rules with simulation and approval; capacity and recalibration history; CI security gate on every change | `change_history.json`; CI `security` job |
| **Integrity of models** | SHA-256 of every model artifact is recorded at registration and verified before loading. Artifacts are pickle files, so a swapped file could otherwise run code | `artifact_hashes.json` per registry family |
| **Secure development** | SAST (bandit: no medium or high findings), dependency audit (pip-audit: 0 known vulnerabilities in the 101 deployable packages), secret scan, all in CI | `security_assessment.json`; `sbom.cyclonedx.json` |
| **Hardening** | Security headers (CSP, frame denial, nosniff, no-referrer, HSTS over HTTPS); API docs off in production; 25 MB request limit; per-principal rate limits; CORS restricted to named origins; `/readyz` reveals only ready / not ready to anonymous callers | Self-assessment TR-1 |
| **Card data minimisation** | PAN tokenised at the adapter (HMAC key from the HSM); track data, PIN block and ICC data dropped; log redaction of PANs and IBANs; PAN discovery scan over database, outputs, logs and model cards. Clean, including a database that processed 200 ISO 8583 card messages | `bti.security.pan_scan`; self-assessment PC-1 |
| **Availability and residency** | Readiness probe, database fallback, replay and rebuild, RPO/RTO targets, India in-country guard (Phase 7) | docs/HA_DR_AND_RESIDENCY.md; `readiness.json`, `residency.json` |
| **Model risk** | Validation workflow, independent sign-offs, findings, fairness gates, outcomes analysis, monitoring (Phases 1–4) | `validation.json` |

**Self-assessment** (`python -m bti.security.assessment`): 12 pass, 2 open.

| Check | Result |
|---|---|
| Every route has an access rule | pass |
| All 141 protected routes refuse anonymous callers | pass |
| Auditor refused on every write route | pass |
| Injection and traversal payloads on every path parameter: no 5xx, no reflection | pass |
| Headers present; docs off | pass |
| SAST | pass |
| Secret scan | pass |
| No card numbers at rest | pass |
| Deployable dependencies free of known vulnerabilities | pass |
| Legacy key off | pass |
| Not in development mode | pass |
| CORS restricted to named origins | pass |
| **Open:** tokenisation key set | per deployment |
| **Open:** residency jurisdiction configured | per deployment |

**Evidence pack** (`python -m bti.security.evidence build | verify`). Ten
system-generated artefacts plus a SHA-256 manifest anchored in the audit log;
`verify` detects any change to a file.

**Fixed during Phase 10:**
- 94 API routes had no authentication beyond the three meant to be public:
  73 reads, including customer transactions and cases, and 21 writes,
  including scoring and model reload.
- Approver and reviewer names were free text behind a single shared key, so
  four-eyes could not be enforced.
- The high-value maker-checker accepted a checker named by the maker.
- Public API documentation, no security headers, no request limits.
- A SHA-1 hash flagged by SAST (an identifier, now marked as such).
- The pipeline's `exec` restricted to an allow-list.

---

## 3. SOC 2 — Trust Services Criteria mapping

| Criterion | BTI contribution | Organisation must provide |
|---|---|---|
| CC1 Control environment | — | Board oversight, org chart, code of conduct, HR screening, training |
| CC2 Communication | Model cards, docket, decision reason codes, customer notices | Security policies communicated; customer commitments (contracts, SLAs) |
| CC3 Risk assessment | Model risk management (validation, findings, fairness), threat model (§6) | Enterprise risk assessment incl. fraud risk; vendor risk |
| CC4 Monitoring of controls | Drift, outcomes, governance checks, early warning, self-assessment in CI | Internal audit; management review of exceptions |
| CC5 Control activities | Access table, four-eyes, maker-checker, validation gates | Policies and procedures approved and owned |
| CC6 Logical access | Principals, RBAC, identity binding, key hashing, deactivation, rate limits, tokenisation | Joiner/mover/leaver process; quarterly access reviews (from `principals_review.json`); **MFA for people**: put human access behind the bank's SSO (gateway as a `system` principal); physical security of hosting |
| CC7 System operations | Hash-chained audit, security logging, alerts, readiness, early warning, PAN scan | Incident response plan and tests; vulnerability management SLAs; SIEM integration |
| CC8 Change management | Immutable registry, gated promotion, rule versioning, CI tests and security gate, evidence of commit per pack | Change advisory approvals; segregation between developers and production deployers |
| CC9 Risk mitigation | DR design, residency guard | Vendor management (cloud, SMS, LLM providers); insurance; business continuity tests |
| A1 Availability | HA/DR design, RPO/RTO targets, readiness, fallback, rebuild | Capacity monitoring; executed DR drills (bank-run, §5 of the HA/DR guide) |
| C1 Confidentiality | Tokenisation, redaction, role-scoped customer data, residency | Data classification, retention and disposal policy |
| PI1 Processing integrity | Point-in-time features, parity tests, idempotent streaming, validation gates, outcomes analysis | Reconciliation sign-off with the incumbent and finance |

**What Type II needs.** These controls must *operate* over the observation
window, with evidence produced throughout. Run the evidence pack on a schedule
(for example monthly) and retain it.

---

## 4. ISO/IEC 27001:2022 — Annex A (Statement of Applicability draft)

Annex A has 93 controls in four themes: organisational (37), people (8),
physical (14) and technological (34). BTI implements or supports most
technological controls. The organisational, people and physical controls are
the operating organisation's ISMS.

| Control | Status | BTI implementation / evidence |
|---|---|---|
| 5.15 Access control, 5.18 Access rights | Implemented (software) | Role table, principals, access reviews from the evidence pack |
| 5.23 Information security for cloud services | Supported | Residency guard, HA/DR design; provider due diligence by organisation |
| 5.28 Collection of evidence | Implemented | Evidence pack with hash manifest anchored in the audit log |
| 5.33 Protection of records | Implemented | Hash-chained audit log, sealed archive, 7-year retention |
| 5.34 Privacy and protection of PII | Supported | Tokenisation, redaction, protected attributes never model inputs, residency |
| 8.2 Privileged access rights | Implemented | `admin` role is distinct; legacy shared admin key off |
| 8.3 Information access restriction | Implemented | Customer-level endpoints restricted to analyst/auditor |
| 8.5 Secure authentication | Partly | Hashed per-principal keys; **MFA for people via SSO gateway: to implement at deployment** |
| 8.7 Protection against malware | Supported | Artifact integrity checks; host anti-malware by organisation |
| 8.8 Management of technical vulnerabilities | Implemented | pip-audit in CI, SBOM, SAST |
| 8.9 Configuration management | Implemented | Versioned settings, config snapshot in evidence, startup refusal of default secrets |
| 8.10 Information deletion, 8.11 Data masking | Implemented | Tokenisation, PAN/IBAN redaction; deletion schedule by organisation |
| 8.12 Data leakage prevention | Partly | Role scoping, PAN scan, residency guard on outbound endpoints |
| 8.15 Logging, 8.16 Monitoring activities | Implemented | Audit chain, security log events (401/403/429), monitoring jobs |
| 8.20–8.22 Network security, segregation | Organisation | Network policy, segmentation (see the PCI section) |
| 8.24 Use of cryptography | Implemented | HMAC-SHA256 tokenisation, SHA-256 key hashing and artifact integrity; TLS at ingress by deployment |
| 8.25–8.29 Secure development lifecycle, coding, testing | Implemented | Tests (300+), CI gates, SAST, dependency audit, security self-assessment |
| 8.31 Separation of environments | Supported | Development-only bypass refused in production; organisation separates environments |
| 8.32 Change management | Implemented | Registry, rules, CI |
| 8.33 Test information | Implemented | Synthetic data only in development |
| 8.34 Protection during audit testing | Implemented | Self-assessment runs against an in-memory database with temporary principals |
| 5.1–5.37 (other organisational), 6.1–6.8 (people), 7.1–7.14 (physical) | Organisation | ISMS policies, roles, supplier management, incident management, HR, physical security of facilities and hosting |

---

## 5. PCI DSS v4.0.1 — scope reduction

**Design goal:** keep BTI out of the CDE.

```
Card switch ──ISO 8583──▶ [ Tokenisation edge: ISO 8583 adapter ]  ──tokens only──▶ Kafka ▶ BTI scoring, DB, store
                         (inside the bank's CDE; HMAC key in HSM;    (out of CDE: no PAN, track, PIN or ICC data)
                          PAN → token, last 4; sensitive data dropped)
```

- **The tokenisation edge.** The ISO 8583 adapter (`bti.streaming.iso8583`)
  is the only BTI component that ever sees a PAN. Deploy it inside the
  bank's CDE, on the switch side of the segmentation boundary, as part of the
  bank's own PCI assessment. Everything downstream receives tokens.
- **Downstream is outside the CDE** if:
  - segmentation is enforced and tested (11.4.5, and every six months for a
    service provider, 11.4.6)
  - the tokenisation key never leaves the HSM / CDE
  - no component can de-tokenise; the HMAC tokens are one-way, with no vault
- **Verification.**
  - `bti.security.pan_scan` provides the yearly scope confirmation evidence
    (12.5.2; every six months for service providers, 12.5.2.1). It has been
    clean on the development system and on a stream database that processed
    card messages.
  - Logs redact PANs as a backstop (3.3, 3.4).

**Requirements that still apply to the tokenisation edge:**
- 3.5.1: the PAN is rendered unreadable (tokenised immediately; never stored)
- 3.6–3.7: key management (the HSM-held key)
- 4.2: TLS on the switch link
- 6.2–6.4: secure development; SAST and CI cover the adapter code
- 8: authentication
- 10: logging
- 11.3–11.4: scans and penetration tests of the edge and the segmentation
- 12.5.2: scope confirmation

**If BTI is offered as a hosted service** and receives PANs itself, the
operator becomes a PCI service provider. Above 300,000 transactions a year
per brand, that means a yearly ROC by a QSA. The tokenisation-edge design
avoids this, and is the recommended deployment.

---

## 6. Threat model (STRIDE)

| Asset / flow | Threat | Mitigation in BTI | Residual / organisation |
|---|---|---|---|
| API keys | **S**poofing: stolen or shared key | Per-principal hashed keys, deactivation, rate limits, legacy key off | MFA via SSO gateway; key rotation; secret manager |
| Approvals (four-eyes) | **S**poofing: approve as someone else | Identity binding of actor fields to the principal | Separate admin from approvers in the identity provider |
| Model artifacts (pickle) | **T**ampering → code execution | SHA-256 recorded at registration, verified on load; registry is append-only | Restrict filesystem / object-store write access; sign artifacts in CI |
| Audit log | **T**ampering / **R**epudiation | Hash chain, DB triggers, sealed archive, evidence anchoring | WORM storage for archives |
| Scoring endpoint | **I**nformation disclosure: model extraction by probing | Authentication, scoring role, rate limit | Gateway quotas; monitor scoring volume per principal |
| Customer data endpoints | **I**nformation disclosure | Analyst/auditor only; no-store caching; residency | Data loss prevention; access reviews |
| Card data | **I**nformation disclosure | Tokenisation at edge, dropped sensitive data, redaction, PAN scan | HSM key custody; segmentation testing |
| ISO 20022 XML | **D**enial of service / XXE | defusedxml, size and count caps | Gateway payload limits |
| API generally | **D**enial of service | 25 MB request limit, rate limits, readiness-based routing | WAF / gateway, autoscaling |
| Copilot (LLM) | **I**nformation disclosure to a third party | Residency guard blocks out-of-country LLM endpoints in enforce mode | Vendor contract (no training on data), redaction policy |
| Roles | **E**levation of privilege | Default-deny table; test that every route has a rule; auditor refused on all writes | Joiner/mover/leaver; quarterly reviews |
| Pipeline `exec` | **E**levation via script injection | Allow-list of stage scripts | Read-only code deployment |

---

## 7. Penetration test: scope and rules of engagement (for the independent tester)

**In scope:**
- the BTI API (all 144 routes; access table in the evidence pack)
- the analyst workbench
- the stream consumer and its Kafka topics
- the ISO 8583 / ISO 20022 adapters, with malformed and hostile messages
- the step-up callback (HMAC)
- the feature store and database network exposure
- authentication and authorisation, including identity binding and
  maker-checker bypass attempts
- model-artifact tampering
- residency-guard bypass
- segmentation between the tokenisation edge and the rest (PCI 11.4.5)

**Test accounts:** one principal per role (scoring, analyst, operations,
model_risk, auditor), created for the test and deactivated after. No
production keys.

**Environment:** a production-like staging deployment with synthetic data. No
real customer or card data.

**Rules:**
- Agreed test windows.
- No denial-of-service against shared infrastructure without approval.
- Stop and report immediately on any critical finding or real data exposure.
- Findings classified CVSS v3.1 / v4.0.
- A retest of fixes is included.

**Before the test:** run `python -m bti.security.assessment` and
`python -m bti.security.evidence build`, and give the tester the access matrix
and this threat model.

**Frequency** (PCI DSS 11.4):
- yearly, and after significant change
- internal and external
- segmentation tests every six months when BTI is operated as a service
  provider

---

## 8. Gaps and roadmap: what the operating organisation must do

| Area | Action | Owner |
|---|---|---|
| ISMS | Scope statement, information security policy, risk assessment and treatment plan, Statement of Applicability (from §4), internal audit, management review | Operating organisation (CISO / founder) |
| People | Background checks, security awareness training, confidentiality agreements, disciplinary process | HR |
| Identity | SSO with MFA in front of BTI for people (gateway as `system` principal); joiner/mover/leaver; quarterly access reviews | IT / security |
| Suppliers | Due diligence and contracts for cloud, SMS/push/3DS providers, LLM provider (no training on data), Kafka/Redis hosting | Procurement / legal |
| Operations | Incident response plan and tabletop test; vulnerability-management SLAs; SIEM ingestion of security logs; backup restore tests; DR drills | Operations |
| Evidence cadence | Monthly evidence packs retained for the observation window; CI history kept | Engineering |
| PCI | Deploy the tokenisation edge in the bank's CDE; HSM key ceremony; segmentation test; scope confirmation with the bank's QSA | Bank + engineering |
| External | SOC 2 readiness assessment → Type I → Type II (observation 3–12 months); ISO Stage 1 → Stage 2; independent penetration test before go-live and yearly | Operating organisation with an auditor, certification body and tester |

**Realistic timeline from today,** for a small operating organisation:
- policies and ISMS foundations: 2–3 months
- penetration test and fixes: alongside
- SOC 2 Type I, or ISO Stage 1: around month 3–4
- SOC 2 Type II report, or ISO certificate: around month 9–12

This depends heavily on the organisation, and the auditors set the actual dates.
