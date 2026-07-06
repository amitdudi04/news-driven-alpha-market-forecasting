import sys
sys.path.append("g:\\News-Driven Alpha Forecasting Chinese Equity Markets using Financial Sentiment")
import app
res = app.run_live_pipeline()
h = res.get('health', {})
print("Confidence:", h.get('confidence', 'error'))
print("Signal:", h.get('signal', 'error'))
print("Explanation:", h.get('explanation_text', 'error'))
print("Dir_prob:", h.get('dir_prob', 'error'))
print("Meta_prob:", h.get('meta_prob', 'error'))
print("Regime:", h.get('regime_impact', 'error'))
