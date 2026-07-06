import os
import pandas as pd
import joblib
import numpy as np

print("=====================================================")
print("PHASE 1 — FILE SYSTEM & ARTIFACT AUDIT")
files_to_check = {
    'data/final_dataset.csv': 'data/final_dataset.csv',
    'data/sentiment_features.csv': 'data/sentiment_features.csv',
    'data/news_daily.csv': 'data/news_daily.csv',
    'data/csi300_features.csv': 'data/csi300_features.csv',
    'models/model_live.pkl': 'models/model_live.pkl',
    'outputs/live_tracking.csv': 'outputs/live_tracking.csv',
    'logs/full_audit_log.csv': 'logs/full_audit_log.csv'
}

cwd = os.getcwd()
for name, rel_path in files_to_check.items():
    p = os.path.join(cwd, rel_path)
    if os.path.exists(p):
        size = os.path.getsize(p)
        print(f"✔ PASS : {name} exists ({size} bytes)")
        if p.endswith('.csv'):
            try:
                df = pd.read_csv(p)
                print(f"  -> Rows: {len(df)}, Cols: {len(df.columns)}")
            except:
                pass
    else:
        print(f"❌ FAIL : {name} DOES NOT EXIST")

print("\n=====================================================")
print("PHASE 4 — MODEL LOADING & INFERENCE CHECK")
model_path = os.path.join(cwd, 'models', 'model_live.pkl')
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        print("✔ PASS : model_live.pkl loaded")
        keys = list(model.keys()) if isinstance(model, dict) else []
        print(f"  -> Keys found: {keys}")
        
        # Test Inference
        if 'feature_cols' in model:
            X_dummy = pd.DataFrame([np.zeros(len(model['feature_cols']))], columns=model['feature_cols'])
            prob_high = model['model_high'].predict_proba(X_dummy)[0][1]
            prob_low = model.get('model_low', model['model_high']).predict_proba(X_dummy)[0][1]
            X_meta = X_dummy.copy()
            X_meta['direction_prob'] = prob_high
            meta_prob = model['meta_model'].predict_proba(X_meta)[0][1]
            print(f"  -> Dummy High Vol Dir Prob: {prob_high}")
            print(f"  -> Dummy Meta Prob: {meta_prob}")
    except Exception as e:
        print(f"❌ FAIL : Model load error: {e}")
else:
    print("❌ FAIL : model_live.pkl missing")

print("\n=====================================================")
print("PHASE 8 — LIVE TRACKING (PnL ENGINE)")
tpath = os.path.join(cwd, 'outputs', 'live_tracking.csv')
if os.path.exists(tpath):
    df = pd.read_csv(tpath)
    print(f"Columns: {list(df.columns)}")
    print(f"Nulls: {df.isnull().sum().sum()}")
    print("Recent rows:")
    print(df[['date', 'signal', 'confidence', 'actual_return', 'pnl']].tail(3))
else:
    print("❌ FAIL : live_tracking.csv missing")
