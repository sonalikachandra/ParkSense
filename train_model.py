"""
ParkSense — Step 2: Train XGBoost Risk Classifier
Run after preprocess.py. Outputs models/xgb_model.pkl and supporting files.
"""
import pandas as pd, numpy as np, xgboost as xgb, pickle, json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

print("⏳ Loading cleaned data...")
df = pd.read_csv("data/cleaned.csv", low_memory=False)
print(f"   Rows: {len(df):,}")

# ── Aggregate per zone × hour × day slot ────────────────────────────────
print("⏳ Building training slots...")
agg = df.groupby(['h3_cell','hour','day_of_week','month',
                  'is_weekend','is_peak_morning','is_peak_evening']).agg(
    violation_count   = ('id', 'count'),
    cell_total        = ('cell_total', 'first'),
    cell_severe_ratio = ('cell_severe_ratio', 'first'),
    cell_dark_ratio   = ('cell_dark_ratio', 'first'),
    cell_repeat_ratio = ('cell_repeat_ratio', 'first'),
).reset_index()
print(f"   Total slots: {len(agg):,}")

# ── Target: top 25% = High Risk ──────────────────────────────────────────
threshold = int(agg['violation_count'].quantile(0.75))
print(f"   High-risk threshold (75th pct): {threshold} violations/slot")
agg['high_risk'] = (agg['violation_count'] >= threshold).astype(int)
print(f"   Class balance — Low: {(agg['high_risk']==0).sum():,}  High: {(agg['high_risk']==1).sum():,}")

with open("models/threshold.json","w") as f:
    json.dump({"threshold": threshold}, f)

# ── Encode H3 cell ────────────────────────────────────────────────────────
le = LabelEncoder()
agg['h3_encoded'] = le.fit_transform(agg['h3_cell'])
with open("models/label_encoder.pkl","wb") as f: pickle.dump(le, f)

# ── Features & split ─────────────────────────────────────────────────────
FEATURES = ['hour','day_of_week','month','is_weekend',
            'is_peak_morning','is_peak_evening','h3_encoded',
            'cell_total','cell_severe_ratio','cell_dark_ratio','cell_repeat_ratio']
with open("models/features.json","w") as f: json.dump({"features": FEATURES}, f)

X = agg[FEATURES]
y = agg['high_risk']
X_train,X_test,y_train,y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ── Train ─────────────────────────────────────────────────────────────────
print("⏳ Training XGBoost...")
model = xgb.XGBClassifier(
    n_estimators=150, max_depth=4, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric='logloss', random_state=42, verbosity=0)
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Low Risk','High Risk']))
print(f"   ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")

print("\n📊 Feature Importance:")
for k,v in sorted(zip(FEATURES, model.feature_importances_), key=lambda x:-x[1]):
    print(f"   {'█'*int(v*40):<40} {k}  {v:.3f}")

with open("models/xgb_model.pkl","wb") as f: pickle.dump(model, f)
print("\n✅ Saved: models/xgb_model.pkl")

# ── Risk grid: predict all cells × 24h × 7 days ──────────────────────────
print("⏳ Generating 24h risk grid for all zones...")
h3_summary = pd.read_csv("data/h3_summary.csv")
rows = []
for _, row in h3_summary.iterrows():
    try: h3_enc = le.transform([row['h3_cell']])[0]
    except: continue
    for hour in range(24):
        for dow in range(7):
            feat = [[hour, dow, 1, int(dow>=5),
                     int(hour in [7,8,9]), int(hour in [17,18,19,20]),
                     h3_enc, row['total_violations'],
                     row['severe_ratio'], row['dark_zone_ratio'], row['repeat_ratio']]]
            prob = model.predict_proba(feat)[0][1]
            rows.append({'h3_cell':row['h3_cell'],'hour':hour,'day_of_week':dow,
                         'risk_probability':round(prob,3),'high_risk':int(prob>=0.5)})

pd.DataFrame(rows).to_csv("data/risk_grid.csv", index=False)
print("✅ Saved: data/risk_grid.csv")
print("\n▶ Next: run  streamlit run app.py")