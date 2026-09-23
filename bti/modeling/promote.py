"""
Assign a registered model to champion or challenger from the command line.
Applies the same controls as POST /api/v1/governance/models/{id}/promote:
passed validation for champion, approver ≠ developer, written rationale.

Usage:
  python -m bti.modeling.promote --list
  python -m bti.modeling.promote --model bti-v3-hgb-20260923160241 --role champion \\
      --approver "jane.smith (Model Risk)" --rationale "Independent validation completed, ref MRM-2026-014"
"""

import argparse
import sys

from bti.modeling import registry


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
    try:
        event = registry.assign_role(args.model, args.role, args.approver, args.rationale)
    except registry.RegistryError as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 1
    print(f"{args.model} is now {args.role} (previous: {event['previous']}), approved by {event['approver']}")
    print("Running API workers pick this up on the next request; POST /api/v1/score/reload-models clears caches.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
