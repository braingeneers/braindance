# ranking.py

**Path:** `braindance/examples/paper_cartpole/ranking.py`
**Module:** `braindance.examples.paper_cartpole.ranking`
**Feature Area:** `Examples and Workshop`
**Entry point:** no — library or imported component

## Overview
Original ranked_pairs metric; source pinned in examples/README.md.

## Connections
- **Used by:** `braindance.examples.1_cartpole` — import consumer hint; not a proven runtime call.

## Dependencies
- `numpy` — external or unresolved local import; source import evidence.

## Classes
None

## Functions
### `find_connectivity_patterns(conn_matrix, high_threshold=1.5, low_threshold=-0.5, w_ab=1.0, w_cd=1.0, w_ad=-1.0, w_cb=-1.0, w_ac=-0.3, w_ca=-0.3)`
> Z-scores connectivity columns, searches distinct directed connection pairs with high/low constraints, scores complementary patterns and similarity, and sorts the resulting tuples by score.
> **Called by:** braindance/examples/1_cartpole.py:46 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/examples/paper_cartpole/ranking.py:4`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| `high_threshold=1.5`, `low_threshold=-0.5`; six scoring weights default to w_ab=1, w_cd=1, w_ad=-1, w_cb=-1, w_ac=-.3, w_ca=-.3. |

## Data Shapes
- Input is a square connectivity matrix; returns descending `(a,b,c,d,score,similarity)` tuples of node indices and scores.

## Notes
None
