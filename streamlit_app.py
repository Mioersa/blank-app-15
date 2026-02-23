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
    if "totalTurnover" not in df.columns or "volume" not in df.columns:
        st.error("❌ Columns 'totalTurnover' and 'volume' not found.")
        st.stop()

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

final_df = pd.concat(filtered).sort_values(["contract", "timestamp"]).reset_index(drop=True)

st.subheader(f"🧾 Combined Data for expiry = {selected_expiry}")
st.dataframe(final_df)

# ------------------------------------------------------------
# Helper functions
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
        return "🟢 Rising"
    elif x < 0:
        return "🔴 Falling"
    else:
        return "⚪ Flat"

def build_summary(col_name):
    """Generic builder that replicates full turnover computation for any column."""
    recs = []
    for lbl in upload_labels:
        sub = final_df[final_df["label"] == lbl]
        if sub.empty:
            continue
        val = sub[col_name].iloc[0]
        price = sub["lastPrice"].iloc[-1] if "lastPrice" in sub else np.nan
        recs.append({"time": lbl, col_name: val, "last_price": price})

    df = pd.DataFrame(recs)
    delta = f"Δ {col_name}"
    df[delta] = df[col_name].diff()
    df["Δ Price"] = df["last_price"].diff()

    # original indicator pipeline
    df["OBT"] = (np.sign(df["Δ Price"].fillna(0)) * df[delta].fillna(0)).cumsum()
    mean, std = df[delta].mean(), df[delta].std()
    df["spike_flag"] = df[delta] > (mean + 2 * std)
    df["Cum"] = df[delta].cumsum()

    rolling_N = 5
    df["RTR"] = df[delta] / df[delta].rolling(rolling_N, min_periods=1).mean()

    ema_short = df[delta].ewm(span=3, adjust=False).mean()
    ema_long = df[delta].ewm(span=10, adjust=False).mean()
    df["TurnOsc"] = ema_short - ema_long
    df["RollCorr"] = df["Δ Price"].rolling(5).corr(df[delta])
    df["RTR_Signal"] = df["RTR"].apply(describe_rtr)
    df["TurnOsc_Signal"] = df["TurnOsc"].apply(describe_osc)
    return df

# ------------------------------------------------------------
# 5. Turnover Indicators Table
# ------------------------------------------------------------
turn_df = build_summary("totalTurnover")
st.subheader("📊 Turnover & Price Indicators Summary")
st.dataframe(
    turn_df[
        [
            "time", "Δ totalTurnover", "Δ Price",
            "RTR", "RTR_Signal", "TurnOsc", "TurnOsc_Signal",
            "OBT", "spike_flag", "RollCorr"
        ]
    ]
)

# ----- Plot selector for Turnover -----
turn_col = st.selectbox(
    "Select metric to plot (Turnover):",
    ["Δ totalTurnover", "RTR", "TurnOsc", "OBT", "RollCorr"],
    key="turnover_plot"
)
fig_turn = go.Figure()
fig_turn.add_trace(go.Scatter(x=turn_df["time"],
                              y=turn_df[turn_col],
                              mode="lines+markers",
                              line=dict(color="orange"),
                              name=turn_col))
fig_turn.update_layout(title=f"Turnover: {turn_col} vs Time",
                       xaxis_title="Time",
                       yaxis_title=turn_col,
                       hovermode="x unified")
st.plotly_chart(fig_turn, use_container_width=True)

# ------------------------------------------------------------
# 6. Volume Indicators Table (same computations)
# ------------------------------------------------------------
vol_df = build_summary("volume")
st.subheader("📊 Volume & Price Indicators Summary")
st.dataframe(
    vol_df[
        [
            "time", "Δ volume", "Δ Price",
            "RTR", "RTR_Signal", "TurnOsc", "TurnOsc_Signal",
            "OBT", "spike_flag", "RollCorr"
        ]
    ]
)

# ----- Plot selector for Volume -----
vol_col = st.selectbox(
    "Select metric to plot (Volume):",
    ["Δ volume", "RTR", "TurnOsc", "OBT", "RollCorr"],
    key="volume_plot"
)
fig_vol = go.Figure()
fig_vol.add_trace(go.Scatter(x=vol_df["time"],
                             y=vol_df[vol_col],
                             mode="lines+markers",
                             line=dict(color="teal"),
                             name=vol_col))
fig_vol.update_layout(title=f"Volume: {vol_col} vs Time",
                      xaxis_title="Time",
                      yaxis_title=vol_col,
                      hovermode="x unified")
st.plotly_chart(fig_vol, use_container_width=True)

# ------------------------------------------------------------
# 7. Legacy charts & combined logic (from your reference)
# ------------------------------------------------------------
axis_type = st.radio("Δ Turnover Y‑axis scale", ["linear", "log"], horizontal=True, key="yaxis_scale")

fig1 = go.Figure()
fig1.add_trace(go.Bar(x=turn_df["time"], y=turn_df["Δ totalTurnover"],
                      name="Δ Turnover", marker_color="orange", opacity=0.6, yaxis="y2"))
fig1.add_trace(go.Scatter(x=turn_df.loc[turn_df["spike_flag"], "time"],
                          y=turn_df.loc[turn_df["spike_flag"], "Δ totalTurnover"],
                          mode="markers", marker=dict(color="red", size=10, symbol="diamond"),
                          name="Spike (>2σ)", yaxis="y2"))
fig1.add_trace(go.Scatter(x=turn_df["time"], y=turn_df["OBT"],
                          mode="lines", line=dict(color="green", width=2, dash="dot"),
                          name="OBT", yaxis="y2"))
fig1.add_trace(go.Scatter(x=turn_df["time"], y=turn_df["last_price"],
                          mode="lines+markers", line=dict(color="blue"),
                          name="Last Price", yaxis="y1"))
fig1.update_layout(
    height=700,
    margin=dict(l=60, r=40, t=60, b=60),
    xaxis=dict(title="Time", rangeslider=dict(visible=True)),
    yaxis=dict(domain=[0.45, 1.0], title="Last Price"),
    yaxis2=dict(domain=[0.0, 0.35], title="Δ Turnover / OBT", type=axis_type),
    title=f"Chart 1 – Last Price and Δ Turnover with OBT & Spikes — Expiry {selected_expiry}",
    legend=dict(orientation="h"),
    hovermode="x unified"
)
st.plotly_chart(fig1, use_container_width=True)

# ------------------------------------------------------------
# 8. Combined Signal + Overall
# ------------------------------------------------------------
def classify(row):
    slope = np.sign(row["Δ totalTurnover"]) or 0
    corr = np.sign(row["RollCorr"]) if not np.isnan(row["RollCorr"]) else 0
    score = slope * corr
    return 1 if score > 0 else (-1 if score < 0 else 0)

turn_df["Signal_Val"] = turn_df.apply(classify, axis=1)
turn_df["Signal_Label"] = turn_df["Signal_Val"].map({
    1: "🟢 Bullish Turnover",
    0: "⚪ Neutral",
    -1: "🔴 Bearish Turnover"
})
turn_df["SMA_ΔTT"] = turn_df["Δ totalTurnover"].rolling(5, min_periods=1).mean().round(2)

st.subheader("🧠 Turnover Behavior Insights (Per Interval)")
st.dataframe(turn_df[["time", "Δ totalTurnover", "SMA_ΔTT", "Δ Price", "RollCorr", "Signal_Label"]])

st.subheader("🪄 Combined Signal Summary")
combo = turn_df[["time", "RTR_Signal", "TurnOsc_Signal", "Signal_Label"]].copy()
st.dataframe(combo)

# simple chart for SMA turnover
fig4 = go.Figure()
fig4.add_trace(go.Bar(x=turn_df["time"], y=turn_df["SMA_ΔTT"],
                      name="SMA Δ Turnover", marker_color="teal", opacity=0.6, yaxis="y2"))
fig4.add_trace(go.Scatter(x=turn_df["time"], y=turn_df["last_price"],
                          mode="lines+markers", line=dict(color="blue"),
                          name="Last Price", yaxis="y1"))
fig4.update_layout(height=600,
                   margin=dict(l=60, r=40, t=60, b=60),
                   xaxis=dict(title="Time", rangeslider=dict(visible=True)),
                   yaxis=dict(domain=[0.45, 1.0], title="Last Price"),
                   yaxis2=dict(domain=[0.0, 0.35], title="SMA Δ Turnover", type=axis_type),
                   title="Chart 4 – Last Price vs SMA(Δ Turnover)",
                   legend=dict(orientation="h"),
                   hovermode="x unified")
st.plotly_chart(fig4, use_container_width=True)

# overall signal
last_rows = turn_df.tail(5)
slope = np.sign(last_rows["Cum"].iloc[-1] - last_rows["Cum"].iloc[0])
corr_sign = np.sign(turn_df["RollCorr"].iloc[-1])
score = slope * corr_sign
if score > 0:
    overall_signal = "🟢 **Bullish Accumulation in Turnover**"
elif score < 0:
    overall_signal = "🔴 **Bearish Distribution in Turnover**"
else:
    overall_signal = "⚪ **Neutral / Indecisive Turnover flow**"

st.subheader("📈 Overall Auto‑Signal")
st.markdown(overall_signal)


