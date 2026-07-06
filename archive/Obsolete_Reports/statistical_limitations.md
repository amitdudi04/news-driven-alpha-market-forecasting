# Statistical Limitations

- Total Dataset Size: 45 days
- Training Size: 36 days
- Validation + Testing Size (Walk-Forward): 9 days

## Known Limitations
- The external dataset was limited to 104 trading days due to GDELT IP bans.
- Walk-forward sample size (21 days) is statistically extremely weak.
- This sample size is insufficient to mathematically prove out-of-sample Alpha.

## What CAN be concluded
- The pipeline infrastructure functions deterministically.
- Training and walk-forward code is technically sound.

## What CANNOT be concluded
- The strategy generates reliable edge.
- The Sharpe ratio over this 21-day period will persist.