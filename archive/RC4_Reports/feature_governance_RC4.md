# Feature Governance (RC4)
- **Training Path**: Strict chronological cutoff with target `dropna()`. Validated via codebase review (Module 4).
- **Inference Path**: Forward fill + Outer Join. `target_return_t+1` safely isolated.
- **Leakage**: ZERO LOOK-AHEAD BIAS. Shift operations occur before split.

**Status:** PASS
