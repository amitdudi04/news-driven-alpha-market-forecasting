# Potential Interview Questions & Answers
**Q: How do you prevent overfitting in your NLP models?**
A: I implemented strict chronological walk-forward validation with expanding windows. The scaler is fit exclusively on the training partition, and hyperparameter grids were heavily constrained (max_depth=3).

**Q: Tell me about your risk management architecture.**
A: The system features a hardcoded SAFE MODE. If any live data feature drifts beyond a Z-Score of 15, or if data is stale by > 1 day, the system throws an execution halt rather than generating a blind prediction.
