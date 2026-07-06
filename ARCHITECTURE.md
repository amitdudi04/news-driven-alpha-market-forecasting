# Architecture
The platform is built on an **Immutable Execution Paradigm**. 
- **SAFE MODE Authority**: The execution engine intercepts all outputs and halts trading if distribution bounds (Z-score > 15) are breached.
- **Regime Switching**: Models bifurcate inference based on median market volatility.
