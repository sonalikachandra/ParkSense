"""
ParkSense — Step 1: Data Preprocessing
Run this FIRST. Place your CSV at:  data/dataset.csv
"""
import pandas as pd, json, h3, numpy as np, os
from sklearn.preprocessing import LabelEncoder

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("⏳ Loading dataset...")
df = pd.read_csv("data/dataset.csv")
print(f"   Raw rows: {len(df):,}")

# ── 1. Filter approved only ──────────────────────────────────────────────
df = df[df['validation_status'] == 'approved'].copy()
print(f"   Approved rows: {len(df):,}")

# ── 2. Parse datetimes ───────────────────────────────────────────────────
df['created_datetime'] = pd.to_datetime(df['created_datetime'], format='mixed', utc=True)

# ── 3. Time features ─────────────────────────────────────────────────────
df['hour']             = df['created_datetime'].dt.hour
df['day_of_week']      = df['created_datetime'].dt.dayofweek
df['day_name']         = df['created_datetime'].dt.day_name()
df['month']            = df['created_datetime'].dt.month
df['month_name']       = df['created_datetime'].dt.strftime('%B')
df['is_weekend']       = df['day_of_week'].isin([5,6]).astype(int)
df['is_peak_morning']  = df['hour'].isin([7,8,9]).astype(int)
df['is_peak_evening']  = df['hour'].isin([17,18,19,20]).astype(int)
df['is_delivery_window'] = df['hour'].isin([10,11,12,13,14]).astype(int)

# ── 4. Parse violation_type JSON ─────────────────────────────────────────
def parse_vtype(v):
    try:    return json.loads(v)
    except: return []

df['violation_list'] = df['violation_type'].apply(parse_vtype)
df['num_violations'] = df['violation_list'].apply(len)

SEVERE = ['WRONG PARKING','PARKING IN A MAIN ROAD',
          'PARKING NEAR ROAD CROSSING','DOUBLE PARKING',
          'PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS']
df['is_severe'] = df['violation_list'].apply(
    lambda x: int(any(s in x for s in SEVERE)))

# ── 5. Dark zone flag ────────────────────────────────────────────────────
df['is_dark_zone'] = (
    df['junction_name'].isna() | (df['junction_name'] == 'No Junction')
).astype(int)
print(f"   Dark zone violations: {df['is_dark_zone'].sum():,} ({df['is_dark_zone'].mean()*100:.1f}%)")

# ── 6. H3 spatial indexing (resolution 7) ───────────────────────────────
print("⏳ Assigning H3 cells (~30 seconds)...")
df['h3_cell'] = [h3.latlng_to_cell(lat, lng, 7)
                 for lat, lng in zip(df['latitude'], df['longitude'])]
print(f"   Unique H3 cells: {df['h3_cell'].nunique()}")

# ── 7. Repeat offender flag ──────────────────────────────────────────────
vc = df['vehicle_number'].value_counts()
repeat_set = set(vc[vc >= 5].index)
df['is_repeat_offender'] = df['vehicle_number'].isin(repeat_set).astype(int)
print(f"   Repeat offenders (5+): {len(repeat_set):,}  |  (10+): {(vc>=10).sum():,}")

# ── 8. Cell-level stats (ML features) ────────────────────────────────────
cell_stats = df.groupby('h3_cell').agg(
    cell_total        = ('id', 'count'),
    cell_severe_ratio = ('is_severe', 'mean'),
    cell_dark_ratio   = ('is_dark_zone', 'mean'),
    cell_repeat_ratio = ('is_repeat_offender', 'mean'),
    cell_lat          = ('latitude', 'mean'),
    cell_lng          = ('longitude', 'mean'),
).reset_index()
df = df.merge(cell_stats, on='h3_cell', how='left')

# ── 9. Save cleaned CSV ───────────────────────────────────────────────────
df.to_csv("data/cleaned.csv", index=False)
print(f"✅ Saved: data/cleaned.csv  ({len(df):,} rows)")

# ── 10. H3 summary for map ────────────────────────────────────────────────
h3_summary = df.groupby('h3_cell').agg(
    total_violations  = ('id', 'count'),
    severe_ratio      = ('is_severe', 'mean'),
    dark_zone_ratio   = ('is_dark_zone', 'mean'),
    repeat_ratio      = ('is_repeat_offender', 'mean'),
    peak_hour         = ('hour', lambda x: x.mode()[0]),
    peak_day          = ('day_name', lambda x: x.mode()[0]),
    top_vehicle_type  = ('vehicle_type', lambda x: x.mode()[0]),
    lat               = ('latitude', 'mean'),
    lng               = ('longitude', 'mean'),
).reset_index()
h3_summary.to_csv("data/h3_summary.csv", index=False)

# ── 11. H3 geo boundaries for hexagon polygons ───────────────────────────
print("⏳ Computing H3 boundaries...")
h3_geo_rows = []
for cell in df['h3_cell'].unique():
    boundary  = h3.cell_to_boundary(cell)
    sub       = df[df['h3_cell'] == cell]
    dark_sub  = sub[sub['is_dark_zone'] == 1]
    h3_geo_rows.append({
        'h3_cell'   : cell,
        'lat'       : sub['latitude'].mean(),
        'lng'       : sub['longitude'].mean(),
        'total'     : len(sub),
        'dark_count': len(dark_sub),
        'severe_ratio': sub['is_severe'].mean(),
        'peak_hour' : sub['hour'].mode()[0],
    })
pd.DataFrame(h3_geo_rows).to_csv("data/h3_geo.csv", index=False)

# ── 12. Junction hourly ───────────────────────────────────────────────────
junc_hourly = df.groupby(['junction_name','hour']).size().reset_index(name='count')
junc_hourly.to_csv("data/junction_hourly.csv", index=False)

# ── 13. Station summary ───────────────────────────────────────────────────
station_sum = df.groupby('police_station').agg(
    total_violations = ('id', 'count'),
    severe_count     = ('is_severe', 'sum'),
    dark_zone_count  = ('is_dark_zone', 'sum'),
    repeat_offenders = ('is_repeat_offender', 'sum'),
    peak_hour        = ('hour', lambda x: x.mode()[0]),
).reset_index().sort_values('total_violations', ascending=False)
station_sum.to_csv("data/station_summary.csv", index=False)

# ── 14. Repeat offender table ────────────────────────────────────────────
repeat_tbl = df[df['is_repeat_offender']==1].groupby('vehicle_number').agg(
    total_violations = ('id', 'count'),
    vehicle_type     = ('vehicle_type', lambda x: x.mode()[0]),
    top_junction     = ('junction_name', lambda x: x.mode()[0]),
    top_hour         = ('hour', lambda x: x.mode()[0]),
    top_violation    = ('violation_list', lambda x: x.mode()[0] if len(x)>0 else ''),
).reset_index().sort_values('total_violations', ascending=False)
repeat_tbl.to_csv("data/repeat_table.csv", index=False)

# ── 15. Junction stats for scheduler ─────────────────────────────────────
print("⏳ Computing junction stats...")
junc_stats = df.groupby('junction_name').agg(
    total           = ('id', 'count'),
    severe_ratio    = ('is_severe', 'mean'),
    repeat_ratio    = ('is_repeat_offender', 'mean'),
    peak_hour       = ('hour', lambda x: x.mode()[0]),
    peak_day        = ('day_name', lambda x: x.mode()[0]),
    lat             = ('latitude', 'mean'),
    lng             = ('longitude', 'mean'),
).reset_index()

def peak_conc(group):
    ph = group['hour'].mode()[0]
    return (group['hour'] == ph).sum() / len(group)

pc = df.groupby('junction_name').apply(peak_conc, include_groups=False).reset_index()
pc.columns = ['junction_name', 'peak_concentration']
junc_stats = junc_stats.merge(pc, on='junction_name')
junc_stats.to_csv("data/junc_stats.csv", index=False)

print("\n✅ All data files saved:")
for f in ["cleaned.csv","h3_summary.csv","h3_geo.csv","junction_hourly.csv",
          "station_summary.csv","repeat_table.csv","junc_stats.csv"]:
    path = f"data/{f}"
    size = os.path.getsize(path) / 1024
    print(f"   {path}  ({size:.0f} KB)")
print("\n▶ Next: run  python train_model.py")