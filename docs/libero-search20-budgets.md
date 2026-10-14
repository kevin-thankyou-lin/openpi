# Prospective LIBERO search20 accounting

Set session budget_mode to exploration20_confirmation10_v1, challenger_cap to20,
confirmation_cap to10, and provide explicit cap_authority. Exploration
(search_candidate and causal_replay) consumes the primary20 cap. Candidate
validation consumes a separate named confirmation10 cap. Baselines retain their
separate registered cap. Failed starts and reservations count as before.

Exploration requires confirmation_opened:false; the live ledger also blocks
exploration once any candidate_validation start is present, regardless of that
flag. Confirmation requires selection_closed:true and a frozen candidate.

Absent budget_mode retains the legacy combined20 behavior. An unknown mode or
ambiguous separate-cap field fails closed. Existing cap records are immutable:
create a reviewed migration and import actual exploration starts, preserving old
confirmation receipts separately. Never reset counts or alter historical ledgers.
This mode adds no rollout authority and changes no actions or simulation physics.

Use STRIDER strider-search20 prompt/validate before prospective dispatch to check
named-parent one-variable changes, repair/acceleration reviews, expansion value,
and the current budget snapshot. Recheck pins and atomically reserve in the live
ledger; an offline admission receipt alone does not authorize execution.
