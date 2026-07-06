# Dashboard Bug Fixes
## Repairs Executed

1. **Pipeline Status Initialization**: Fixed bug in `app.py` where `p_cols[0].metric()` defaulted to ✖ even when health was "OK".
2. **SAFE MODE Desync**: Re-wrote safe mode detection banners to strictly ensure all labels read "ACTIVE" synchronously. Removed duplicate and conflicting logic blocks.
3. **Empty Data Collapses**: Re-configured the `df_plot` graphing limits. Graphs and worst prediction metrics now safely hide behind a `st.warning()` if length < 20 days.
4. **Hardcoded Values**: Scrubbed all instances of hardcoded "CRASHED" architecture blocks. Replaced with actual config/model references.
