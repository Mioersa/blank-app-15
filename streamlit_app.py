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
    type="csv",
    accept_multiple_files=True,
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
    recs = []
    for lbl in upload_labels:
        sub = final_df[final_df["label"] == lbl]
        if sub.empty:
            continue
        val = sub[col_name].iloc[0]
        price = sub["lastPrice"].iloc[-1] if "lastPrice" in sub else np.nan
        recs.append({"time": lbl, col_name: val, "last_price": price})

    df = pd.DataFrame(recs)
    dcol = f"Δ {col_name}"
    df[dcol] = df[col_name].diff()
    df["Δ Price"] = df["last_price"].diff()
    df["OBT"] = (np.sign(df["Δ Price"].fillna(0)) * df[dcol].fillna(0)).cumsum()
    mean, std = df[dcol].mean(), df[dcol].std()
    df["spike_flag"] = df[dcol] > (mean + 2 * std)
    df["Cum"] = df[dcol].cumsum()
    df["RTR"] = df[dcol] / df[dcol].rolling(5, min_periods=1).mean()
    ema_s, ema_l = df[dcol].ewm(span=3, adjust=False).mean(), df[dcol].ewm(span=10, adjust=False).mean()
    df["TurnOsc"] = ema_s - ema_l
    df["RollCorr"] = df["Δ Price"].rolling(5).corr(df[dcol])
    df["RTR_Signal"] = df["RTR"].apply(describe_rtr)
    df["TurnOsc_Signal"] = df["TurnOsc"].apply(describe_osc)
    return df

# ------------------------------------------------------------
# Turnover Indicators Table
# ------------------------------------------------------------
turn_df = build_summary("totalTurnover")
st.subheader("📊 Turnover & Price Indicators Summary")
st.dataframe(turn_df[[ "time","Δ totalTurnover","Δ Price","RTR","RTR_Signal","TurnOsc","TurnOsc_Signal","OBT","spike_flag","RollCorr"]])

turn_sel = st.selectbox("Select metric to plot (Turnover):",
                        ["Δ totalTurnover","RTR","TurnOsc","OBT","RollCorr"], key="turn_sel")
fig_turn = go.Figure([go.Scatter(x=turn_df["time"], y=turn_df[turn_sel],
                                 mode="lines+markers", line=dict(color="orange"))])
fig_turn.update_layout(title=f"Turnover: {turn_sel} vs Time", xaxis_title="Time", yaxis_title=turn_sel)
st.plotly_chart(fig_turn, use_container_width=True)

# ------------------------------------------------------------
# Volume Indicators Table
# ------------------------------------------------------------
vol_df = build_summary("volume")
st.subheader("📊 Volume & Price Indicators Summary")
st.dataframe(vol_df[[ "time","Δ volume","Δ Price","RTR","RTR_Signal","TurnOsc","TurnOsc_Signal","OBT","spike_flag","RollCorr"]])

vol_sel = st.selectbox("Select metric to plot (Volume):",
                       ["Δ volume","RTR","TurnOsc","OBT","RollCorr"], key="vol_sel")
fig_vol = go.Figure([go.Scatter(x=vol_df["time"], y=vol_df[vol_sel],
                                mode="lines+markers", line=dict(color="teal"))])
fig_vol.update_layout(title=f"Volume: {vol_sel} vs Time", xaxis_title="Time", yaxis_title=vol_sel)
st.plotly_chart(fig_vol, use_container_width=True)

# ------------------------------------------------------------
# 🛠️ Customizable Column Analytics
# ------------------------------------------------------------
st.subheader("🛠️ Customizable Column Analytics")

col_choice = st.selectbox("Select a column from combined data to analyze", final_df.columns, key="custom_col")
if st.button("Calculate Custom Summary"):
    recs = []
    for lbl in upload_labels:
        sub = final_df[final_df["label"] == lbl]
        if sub.empty:
            continue
        val = sub[col_choice].iloc[0] if col_choice in sub else np.nan
        last_price = sub["lastPrice"].iloc[-1] if "lastPrice" in sub else np.nan
        recs.append({"time": lbl, col_choice: val, "last_price": last_price})

    customdf = pd.DataFrame(recs)
    dcol = f"Δ {col_choice}"
    customdf[dcol] = customdf[col_choice].diff()
    customdf["Δ Price"] = customdf["last_price"].diff()
    customdf["OBT"] = (np.sign(customdf["Δ Price"].fillna(0)) * customdf[dcol].fillna(0)).cumsum()
    m, s = customdf[dcol].mean(), customdf[dcol].std()
    customdf["spike_flag"] = customdf[dcol] > (m + 2 * s)
    customdf["RTR"] = customdf[dcol] / customdf[dcol].rolling(5, min_periods=1).mean()
    ema_s, ema_l = customdf[dcol].ewm(span=3, adjust=False).mean(), customdf[dcol].ewm(span=10, adjust=False).mean()
    customdf["TurnOsc"] = ema_s - ema_l
    customdf["RollCorr"] = customdf["Δ Price"].rolling(5).corr(customdf[dcol])
    customdf["RTR_Signal"] = customdf["RTR"].apply(describe_rtr)
    customdf["TurnOsc_Signal"] = customdf["TurnOsc"].apply(describe_osc)

    st.subheader(f"📊 Customised — {col_choice} & Price Indicators Summary")
    st.dataframe(customdf[[ "time", dcol, "Δ Price","RTR","RTR_Signal","TurnOsc","TurnOsc_Signal","OBT","spike_flag","RollCorr"]])

    sel_metric = st.selectbox("Select column to plot (Custom):",
                              [dcol,"RTR","TurnOsc","OBT","RollCorr"], key="custom_sel")
    fig_custom = go.Figure([go.Scatter(x=customdf["time"], y=customdf[sel_metric],
                                       mode="lines+markers", line=dict(color="purple"))])
    fig_custom.update_layout(title=f"{col_choice}: {sel_metric} vs Time",
                             xaxis_title="Time", yaxis_title=sel_metric)
    st.plotly_chart(fig_custom, use_container_width=True)

# ------------------------------------------------------------
# Legacy combined chart & overall signal
# ------------------------------------------------------------
axis_type = st.radio("Δ Turnover Y‑axis scale", ["linear", "log"], horizontal=True)
fig1 = go.Figure()
fig1.add_bar(x=turn_df["time"], y=turn_df["Δ totalTurnover"], name="Δ Turnover", marker_color="orange", opacity=0.6)
fig1.add_scatter(x=turn_df["time"], y=turn_df["last_price"], mode="lines+markers", name="Last Price", yaxis="y2", line=dict(color="blue"))
fig1.update_layout(title=f"Δ Turnover & Price – Expiry {selected_expiry}",
                   xaxis=dict(title="Time", rangeslider=dict(visible=True)),
                   yaxis2=dict(title="Price", overlaying="y", side="right"),
                   height=600, hovermode="x unified")
st.plotly_chart(fig1, use_container_width=True)

# simple auto-signal
last_rows = turn_df.tail(5)
slope = np.sign(last_rows["Cum"].iloc[-1] - last_rows["Cum"].iloc[0])
corr_sign = np.sign(turn_df["RollCorr"].iloc[-1])
score = slope * corr_sign
overall_signal = "🟢 **Bullish Accumulation**" if score > 0 else ("🔴 **Bearish Distribution**" if score < 0 else "⚪ **Neutral Flow**")
st.subheader("📈 Overall Auto‑Signal")
st.markdown(overall_signal)
