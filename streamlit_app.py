
import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(page_title="Intraday Futures Turnover Analytics", layout="wide")

# ------------------------------------------------------------
# 1. File Upload
# ------------------------------------------------------------
st.sidebar.title("📂 Upload Futures CSVs")
uploaded_files = st.sidebar.file_uploader(
    "Drag‑drop intraday futures CSVs (5‑min interval)",
    type="csv", accept_multiple_files=True,
)

if not uploaded_files:
    st.warning("👋 Upload at least one CSV.")
    st.stop()

# ------------------------------------------------------------
# 2. Read uploaded CSVs
# ------------------------------------------------------------
dfs, upload_times, upload_labels = [], [], []

for uploaded in uploaded_files:
    fn = uploaded.name
    m = re.search(r"_(\d{2})(\d{2})(\d{4})_(\d{2})(\d{2})(\d{2})\.csv$", fn)
    base_time = datetime.now()
    label = "unknown"

    if m:
        dd, mm, yyyy, HH, MM, SS = m.groups()
        base_time = datetime(int(yyyy), int(mm), int(dd), int(HH), int(MM), int(SS))
        label = f"{HH}:{MM}"

    upload_times.append(base_time)
    upload_labels.append(label)

    df = pd.read_csv(uploaded)
    if "totalTurnover" not in df.columns:
        st.error("❌ Column 'totalTurnover' not found.")
        st.stop()

    # round turnover to 2 decimals
    df["totalTurnover"] = df["totalTurnover"].round(2)
    df["timestamp"] = base_time + pd.to_timedelta(np.arange(len(df)) * 5, unit="min")
    dfs.append(df)

# ------------------------------------------------------------
# 3. Expiry selector
# ------------------------------------------------------------
first_df = dfs[0]
if "expiryDate" not in first_df.columns:
    st.error("❌ Column 'expiryDate' not found.")
    st.stop()

expiry_options = sorted(first_df["expiryDate"].unique())
selected_expiry = st.selectbox("Select Expiry Date", expiry_options)

# ------------------------------------------------------------
# 4. Filter by expiry
# ------------------------------------------------------------
filtered = []
for i, df_file in enumerate(dfs):
    sub = df_file[df_file["expiryDate"] == selected_expiry].copy()
    if sub.empty:
        continue
    sub["label"] = upload_labels[i]
    sub["capture_time"] = upload_times[i]
    filtered.append(sub)

if not filtered:
    st.warning("No data for chosen expiry.")
    st.stop()

final_df = pd.concat(filtered).sort_values(
    ["contract", "timestamp"]
).reset_index(drop=True)

st.subheader(f"🧾 Combined Data for expiry = {selected_expiry}")
st.dataframe(final_df)

# ------------------------------------------------------------
# 5. Snapshot summary using totalTurnover
# ------------------------------------------------------------
records = []
for lbl in upload_labels:
    sub = final_df[final_df["label"] == lbl]
    if sub.empty:
        continue
    tt = sub["totalTurnover"].iloc[0]
    price = sub["lastPrice"].iloc[-1] if "lastPrice" in sub else np.nan
    records.append({"time": lbl, "totalTurnover": tt, "last_price": price})

sumdf = pd.DataFrame(records)
sumdf["Δ TT"] = sumdf["totalTurnover"].diff()
sumdf["Δ Price"] = sumdf["last_price"].diff()

# ------------------------------------------------------------
# 6. Indicator calculations (Turnover stats)
# ------------------------------------------------------------
sumdf["OBT"] = (np.sign(sumdf["Δ Price"].fillna(0)) * sumdf["Δ TT"].fillna(0)).cumsum()

tt_mean, tt_std = sumdf["Δ TT"].mean(), sumdf["Δ TT"].std()
sumdf["spike_flag"] = sumdf["Δ TT"] > (tt_mean + 2 * tt_std)
sumdf["Cum_TT"] = sumdf["Δ TT"].cumsum()

rolling_N = 5
sumdf["RTR"] = sumdf["Δ TT"] / sumdf["Δ TT"].rolling(rolling_N, min_periods=1).mean()

short, long = 3, 10
ema_short = sumdf["Δ TT"].ewm(span=short, adjust=False).mean()
ema_long = sumdf["Δ TT"].ewm(span=long, adjust=False).mean()
sumdf["TurnOsc"] = ema_short - ema_long

sumdf["RollCorr"] = sumdf["Δ Price"].rolling(5).corr(sumdf["Δ TT"])

# ------------------------------------------------------------
# 7. Label signals
# ------------------------------------------------------------
def describe_rtr(x):
    if x > 1.5:
        return "🚀 High"
    elif x < 0.7:
        return "🧊 Low"
    else:
        return "⚪ Normal"

def describe_osc(x):
    if x > 0:
        return "🟢 Rising Turnover"
    elif x < 0:
        return "🔴 Falling Turnover"
    else:
        return "⚪ Flat"

sumdf["RTR_Signal"] = sumdf["RTR"].apply(describe_rtr)
sumdf["TurnOsc_Signal"] = sumdf["TurnOsc"].apply(describe_osc)

st.subheader("📊 Turnover & Price Indicators Summary")
st.dataframe(
    sumdf[
        [
            "time", "Δ TT", "Δ Price",
            "RTR", "RTR_Signal",
            "TurnOsc", "TurnOsc_Signal",
            "OBT", "spike_flag", "RollCorr",
        ]
    ]
)

# ------------------------------------------------------------
# 8. Chart 1 — ΔTT + OBT + Spikes + Price
# ------------------------------------------------------------
st.subheader("📈 Chart 1 – Last Price (top) & Δ Turnover (bottom + OBT + Spikes)")
axis_type = st.radio("Δ Turnover Y‑axis scale", ["linear", "log"], horizontal=True, key="yaxis_scale")

fig1 = go.Figure()
fig1.add_trace(go.Bar(x=sumdf["time"], y=sumdf["Δ TT"],
                      name="Δ Turnover", marker_color="orange", opacity=0.6, yaxis="y2"))
fig1.add_trace(go.Scatter(x=sumdf.loc[sumdf["spike_flag"], "time"],
                          y=sumdf.loc[sumdf["spike_flag"], "Δ TT"],
                          mode="markers", marker=dict(color="red", size=10, symbol="diamond"),
                          name="Spike (>2σ)", yaxis="y2"))
fig1.add_trace(go.Scatter(x=sumdf["time"], y=sumdf["OBT"],
                          mode="lines", line=dict(color="green", width=2, dash="dot"),
                          name="OBT", yaxis="y2"))
fig1.add_trace(go.Scatter(x=sumdf["time"], y=sumdf["last_price"],
                          mode="lines+markers", line=dict(color="blue"),
                          name="Last Price", yaxis="y1"))

fig1.update_layout(
    height=700,
    margin=dict(l=60, r=40, t=60, b=60),
    xaxis=dict(title="Capture Time (HH:MM)", rangeslider=dict(visible=True)),
    yaxis=dict(domain=[0.45, 1.0], title="Last Price"),
    yaxis2=dict(domain=[0.0, 0.35], title="Δ Turnover / OBT", type=axis_type),
    title=f"Chart 1 – Last Price and Δ Turnover with OBT & Spikes — Expiry {selected_expiry}",
    legend=dict(orientation="h"),
    hovermode="x unified"
)
st.plotly_chart(fig1, use_container_width=True)

# ------------------------------------------------------------
# 9. Simplified Chart 2
# ------------------------------------------------------------
st.subheader("📉 Chart 2 – Last Price (top) & Δ Turnover (bottom)")
fig2 = go.Figure()
fig2.add_trace(go.Bar(x=sumdf["time"], y=sumdf["Δ TT"],
                      name="Δ Turnover", marker_color="orange", opacity=0.6, yaxis="y2"))
fig2.add_trace(go.Scatter(x=sumdf["time"], y=sumdf["last_price"],
                          mode="lines+markers", line=dict(color="blue"),
                          name="Last Price", yaxis="y1"))
fig2.update_layout(
    height=600,
    margin=dict(l=60, r=40, t=60, b=60),
    xaxis=dict(title="Capture Time (HH:MM)", rangeslider=dict(visible=True)),
    yaxis=dict(domain=[0.45, 1.0], title="Last Price"),
    yaxis2=dict(domain=[0.0, 0.35], title="Δ Turnover", type=axis_type),
    title="Chart 2 – Clean Δ Turnover Chart",
    legend=dict(orientation="h"),
    hovermode="x unified"
)
st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------------------------
# 10. Interval classification + SMA (ΔTT)
# ------------------------------------------------------------
def classify(row):
    slope = np.sign(row["Δ TT"]) or 0
    corr = np.sign(row["RollCorr"]) if not np.isnan(row["RollCorr"]) else 0
    score = slope * corr
    return 1 if score > 0 else (-1 if score < 0 else 0)

sumdf["Signal_Val"] = sumdf.apply(classify, axis=1)
sumdf["Signal_Label"] = sumdf["Signal_Val"].map({1: "🟢 Bullish Turnover", 0: "⚪ Neutral", -1: "🔴 Bearish Turnover"})
sumdf["SMA_ΔTT"] = sumdf["Δ TT"].rolling(5, min_periods=1).mean().round(2)

st.subheader("🧠 Turnover Behavior Insights (Per Interval)")
st.dataframe(sumdf[["time", "Δ TT", "SMA_ΔTT", "Δ Price", "RollCorr", "Signal_Label"]])

# ------------------------------------------------------------
# 11. Combined signals
# ------------------------------------------------------------
st.subheader("🪄 Combined Signal Summary")
combo = sumdf[["time", "RTR_Signal", "TurnOsc_Signal", "Signal_Label"]].copy()
st.dataframe(combo)

# ------------------------------------------------------------
# 12. Chart 4 — Price vs SMA(ΔTT)
# ------------------------------------------------------------
st.subheader("🆕 Chart 4 – Last Price (top) & SMA(Δ Turnover) (bottom)")
fig4 = go.Figure()
fig4.add_trace(go.Bar(
    x=sumdf["time"], y=sumdf["SMA_ΔTT"],
    name="SMA Δ Turnover",
    marker_color="teal", opacity=0.6, yaxis="y2"
))
fig4.add_trace(go.Scatter(
    x=sumdf["time"], y=sumdf["last_price"],
    mode="lines+markers",
    line=dict(color="blue"),
    name="Last Price", yaxis="y1"
))
fig4.update_layout(
    height=600,
    margin=dict(l=60, r=40, t=60, b=60),
    xaxis=dict(title="Capture Time (HH:MM)", rangeslider=dict(visible=True)),
    yaxis=dict(domain=[0.45, 1.0], title="Last Price"),
    yaxis2=dict(domain=[0.0, 0.35], title="SMA Δ Turnover", type=axis_type),
    title="Chart 4 – Last Price vs SMA(Δ Turnover)",
    legend=dict(orientation="h"),
    hovermode="x unified"
)
st.plotly_chart(fig4, use_container_width=True)

# ------------------------------------------------------------
# 13. Overall Directional Signal
# ------------------------------------------------------------
last_rows = sumdf.tail(5)
slope = np.sign(last_rows["Cum_TT"].iloc[-1] - last_rows["Cum_TT"].iloc[0])
corr_sign = np.sign(sumdf["RollCorr"].iloc[-1])
score = slope * corr_sign

if score > 0:
    overall_signal = "🟢 **Bullish Accumulation in Turnover**"
elif score < 0:
    overall_signal = "🔴 **Bearish Distribution in Turnover**"
else:
    overall_signal = "⚪ **Neutral / Indecisive Turnover flow**"

st.subheader("📈 Overall Auto‑Signal")
st.markdown(overall_signal)
