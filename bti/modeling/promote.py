# Copyright (c) 2026 Srijan Upadhyay. All rights reserved. Proprietary — see LICENSE.
"""
Assign a registered model to champion or challenger from the command line.
Applies the same controls as POST /api/v1/governance/models/{id}/promote:
for champion, passed automated gates, an in-date approving sign-off from an
independent validator, no open high-severity findings, approver ≠ developer,
and a written rationale.

Usage:
  python -m bti.modeling.promote --list
  python -m bti.modeling.promote --model bti-v3-lgbm-20260923180905 --role champion \\
      --approver "jane.smith (Model Risk)" --rationale "Independent validation completed, ref MRM-2026-014"
"""

import argparse
import sys

from bti.modeling import registry


def _session():
    from bti.database.connection import SessionLocal
    from bti.database.init_db import create_tables
    create_tables()
    return SessionLocal()


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote a BTI model (four-eyes controlled)")
    parser.add_argument("--list", action="store_true", help="Show registered models and current roles")
    parser.add_argument("--model")
    parser.add_argument("--role", choices=registry.ROLES, default="champion")
    parser.add_argument("--approver")
    parser.add_argument("--rationale")
    args = parser.parse_args()

    index = registry.read_index()
    if args.list or not args.model:
        print(f"champion:   {index.get('champion')}\nchallenger: {index.get('challenger')}\n")
        for m in index["models"]:
            print(f"  {m['model_id']}  validation={m['validation_status']}  developer={m['developer']}")
        return 0
    if not args.approver or not args.rationale:
        parser.error("--approver and --rationale are required to change a role")
    from bti.database.models import AuditLog
    from bti.governance.validation import ValidationError, assert_ready_for_champion
    db = _session()
    try:
        registry.load_card(args.model)
        if args.role == "champion":
            registry.check_four_eyes(args.model, args.approver)
            assert_ready_for_champion(db, args.model)
        event = registry.assign_role(args.model, args.role, args.approver, args.rationale)
        db.add(AuditLog(event_type="MODEL_ROLE_ASSIGNED", payload=event))
        db.commit()
    except (registry.RegistryError, ValidationError) as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 1
    finally:
        db.close()
    print(f"{args.model} is now {args.role} (previous: {event['previous']}), approved by {event['approver']}")
    print("Running API workers pick this up on the next request; POST /api/v1/score/reload-models clears caches.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
