# 05_rl_cartpole_logs.py

**Path:** `braindance/utils/data_manager/advanced_tutorials/05_rl_cartpole_logs.py`
**Module:** `braindance.utils.data_manager.advanced_tutorials.05_rl_cartpole_logs`
**Feature Area:** `Examples and Workshop`
**Entry point:** no — library or imported component

## Overview
Illustrates loading CartPole game, pattern, and reward logs and deriving reward/state/neural-activity metrics. Intended plots compare reward accumulation, pole angle, action distributions, and neural activity.

## Connections
- **Shared data:** load_catalog/load_recording→lazy RL logs→pandas/NumPy derived columns→Styler figures.

## Dependencies
None

## Classes
None

## Functions
None

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| PROJ='24-04-18_butterfly', CHIP='p001237', EXPERIMENT='exp4/exp4_cartpole_long_1'. |

## Data Shapes
- game_log contains time, reward, action, pole_angle, length-4 state, and left/right spike-count arrays; pattern and reward logs are tabular.

## Notes
- Source currently fails AST parsing with SyntaxError: unexpected indent at line 122.
- Workflow assumes optional RL logs exist and derives reward, state, action, and neural-activity metrics.
- SyntaxError at line 122: unexpected indent
