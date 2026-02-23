import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(page_title="Intraday Futures Turnover Analytics", layout="wide")

# ============================================================
# 1. FILE UPLOAD
# ============================================================
st.sidebar.title("📂 Upload Futures CSVs")
uploaded_files = st.sidebar.file_uploader(
    "Drag‑drop intraday futures CSVs (5‑min interval)",
    type="csv",
    accept_multiple_files=True,
)
if not uploaded_files:
    st.warning("👋 Upload at least one CSV.")
    st.stop()

# ============================================================
# 2. READ CSVs
# ============================================================
dfs, upload_times, upload_labels = [], [], []
for uploaded in uploaded_files:
    fn = uploaded.name
    m = re.search(r"(\d{2})(\d{2})(\d{4})(\d{2})(\d{2})(\d{2}).csv$", fn)
    base_time = datetime.now()
    label = "unknown"
    if m:
        dd, mm, yyyy, HH, MM, SS = m.groups()
        base_time = datetime(int(yyyy), int(mm), int(dd), int(HH), int(MM), int(SS))
        label = f"{HH}:{MM}"

    upload_times.append(base_time)
    upload_labels.append(label)

    df = pd.read_csv(uploaded)
    required = {"totalTurnover","volume","noOfTrades"}
    miss = required - set(df.columns)
    if miss:
        st.error(f"❌ Missing columns: {', '.join(miss)}")
        st.stop()

    df["totalTurnover"] = df["totalTurnover"].round(2)
    df["timestamp"] = base_time + pd.to_timedelta(np.arange(len(df))*5,unit="min")
    dfs.append(df)

# ============================================================
# 3. EXPIRY SELECTOR
# ============================================================
first_df = dfs[0]
if "expiryDate" not in first_df.columns:
    st.error("❌ Column 'expiryDate' not found.")
    st.stop()

expiry_options = sorted(first_df["expiryDate"].unique())
selected_expiry = st.selectbox("Select Expiry Date", expiry_options)

# ============================================================
# 4. FILTER BY EXPIRY
# ============================================================
filtered=[]
for i,df_in in enumerate(dfs):
    sub=df_in[df_in["expiryDate"]==selected_expiry].copy()
    if sub.empty: continue
    sub["label"]=upload_labels[i]
    sub["capture_time"]=upload_times[i]
    filtered.append(sub)
if not filtered:
    st.warning("No data for chosen expiry.");st.stop()

final_df=pd.concat(filtered).sort_values(["contract","timestamp"]).reset_index(drop=True)
st.subheader(f"🧾 Combined Data for expiry = {selected_expiry}")
st.dataframe(final_df)

# ============================================================
# HELPERS
# ============================================================
def describe_rtr(x):
    if x>1.5: return "🚀 High"
    elif x<0.7: return "🧊 Low"
    else: return "⚪ Normal"

def describe_osc(x):
    if x>0: return "🟢 Rising"
    elif x<0: return "🔴 Falling"
    else: return "⚪ Flat"

def build_summary(col):
    recs=[]
    for lbl in upload_labels:
        sub=final_df[final_df["label"]==lbl]
        if sub.empty: continue
        val=sub[col].iloc[0]
        price=sub["lastPrice"].iloc[-1] if "lastPrice" in sub else np.nan
        recs.append({"time":lbl,col:val,"last_price":price})
    df=pd.DataFrame(recs)
    d=f"Δ {col}"
    df[d]=df[col].diff()
    df["Δ Price"]=df["last_price"].diff()
    df["OBT"]=(np.sign(df["Δ Price"].fillna(0))*df[d].fillna(0)).cumsum()
    mean,std=df[d].mean(),df[d].std()
    df["spike_flag"]=df[d]>(mean+2*std)
    df["Cum"]=df[d].cumsum()
    df["RTR"]=df[d]/df[d].rolling(5,min_periods=1).mean()
    ema_s=df[d].ewm(span=3,adjust=False).mean()
    ema_l=df[d].ewm(span=10,adjust=False).mean()
    df["TurnOsc"]=ema_s-ema_l
    df["RollCorr"]=df["Δ Price"].rolling(5).corr(df[d])
    df["RTR_Signal"]=df["RTR"].apply(describe_rtr)
    df["TurnOsc_Signal"]=df["TurnOsc"].apply(describe_osc)
    return df

# ============================================================
# 5. TURNOVER TABLE + PLOT
# ============================================================
turn_df=build_summary("totalTurnover")
st.subheader("📊 Turnover & Price Indicators Summary")
st.dataframe(turn_df[["time","Δ totalTurnover","Δ Price","RTR","RTR_Signal","TurnOsc","TurnOsc_Signal","OBT","spike_flag","RollCorr"]])
sel_turn=st.selectbox("Select metric to plot (Turnover):",["Δ totalTurnover","RTR","TurnOsc","OBT","RollCorr"])
st.plotly_chart(go.Figure([go.Scatter(x=turn_df["time"],y=turn_df[sel_turn],mode="lines+markers",name=sel_turn,line=dict(color="orange"))]).update_layout(title=f"{sel_turn} vs Time",xaxis_title="Time",yaxis_title=sel_turn,hovermode="x unified"),use_container_width=True)

# ============================================================
# 6. VOLUME TABLE + PLOT
# ============================================================
vol_df=build_summary("volume")
st.subheader("📊 Volume & Price Indicators Summary")
st.dataframe(vol_df[["time","Δ volume","Δ Price","RTR","RTR_Signal","TurnOsc","TurnOsc_Signal","OBT","spike_flag","RollCorr"]])
sel_vol=st.selectbox("Select metric to plot (Volume):",["Δ volume","RTR","TurnOsc","OBT","RollCorr"])
st.plotly_chart(go.Figure([go.Scatter(x=vol_df["time"],y=vol_df[sel_vol],mode="lines+markers",name=sel_vol,line=dict(color="teal"))]).update_layout(title=f"{sel_vol} vs Time",xaxis_title="Time",yaxis_title=sel_vol,hovermode="x unified"),use_container_width=True)

# ============================================================
# 7. NO. OF TRADES TABLE + PLOT
# ============================================================
trades_df=build_summary("noOfTrades")
st.subheader("📊 Number of Trades & Price Indicators Summary")
st.dataframe(trades_df[["time","Δ noOfTrades","Δ Price","RTR","RTR_Signal","TurnOsc","TurnOsc_Signal","OBT","spike_flag","RollCorr"]])
sel_trades=st.selectbox("Select metric to plot (No of Trades):",["Δ noOfTrades","RTR","TurnOsc","OBT","RollCorr"])
st.plotly_chart(go.Figure([go.Scatter(x=trades_df["time"],y=trades_df[sel_trades],mode="lines+markers",name=sel_trades,line=dict(color="purple"))]).update_layout(title=f"{sel_trades} vs Time",xaxis_title="Time",yaxis_title=sel_trades,hovermode="x unified"),use_container_width=True)

# ============================================================
# 8. CUSTOMIZABLE TABLE
# ============================================================
st.subheader("🛠️ Customizable Column Analytics")
col_choice=st.selectbox("Select a column from combined data",final_df.columns)
if st.button("Calculate Custom Summary"):
    custom=build_summary(col_choice)
    st.subheader(f"📊 Customised — {col_choice} & Price Indicators Summary")
    st.dataframe(custom[["time",f"Δ {col_choice}","Δ Price","RTR","RTR_Signal","TurnOsc","TurnOsc_Signal","OBT","spike_flag","RollCorr"]])
    pick=st.selectbox("Select column to plot (Custom):",[f"Δ {col_choice}","RTR","TurnOsc","OBT","RollCorr"])
    st.plotly_chart(go.Figure([go.Scatter(x=custom["time"],y=custom[pick],mode="lines+markers",name=pick,line=dict(color="gray"))]).update_layout(title=f"{col_choice}: {pick} vs Time",xaxis_title="Time",yaxis_title=pick,hovermode="x unified"),use_container_width=True)

# ============================================================
# 9. CHART 1 – ΔTT + OBT + SPIKES + PRICE
# ============================================================
axis_type=st.radio("Δ Turnover Y‑axis scale",["linear","log"],horizontal=True)
fig1=go.Figure()
fig1.add_trace(go.Bar(x=turn_df["time"],y=turn_df["Δ totalTurnover"],name="Δ Turnover",marker_color="orange",opacity=0.6,yaxis="y2"))
fig1.add_trace(go.Scatter(x=turn_df["time"],y=turn_df["OBT"],mode="lines",line=dict(color="green",dash="dot"),name="OBT",yaxis="y2"))
fig1.add_trace(go.Scatter(x=turn_df["time"],y=turn_df["last_price"],mode="lines+markers",line=dict(color="blue"),name="Last Price",yaxis="y1"))
fig1.update_layout(height=700,xaxis=dict(title="Time",rangeslider=dict(visible=True)),
    yaxis=dict(domain=[0.45,1],title="Last Price"),yaxis2=dict(domain=[0,0.35],title="Δ Turnover / OBT",type=axis_type),
    title=f"Chart 1 – Turnover and Price — Expiry {selected_expiry}",legend=dict(orientation="h"),hovermode="x unified")
st.plotly_chart(fig1,use_container_width=True)

# ============================================================
# 10. SIMPLIFIED CHART 2
# ============================================================
st.subheader("📉 Chart 2 – Last Price (top) & Δ Turnover (bottom)")
fig2=go.Figure()
fig2.add_bar(x=turn_df["time"],y=turn_df["Δ totalTurnover"],name="Δ Turnover",marker_color="orange",opacity=0.6,yaxis="y2")
fig2.add_scatter(x=turn_df["time"],y=turn_df["last_price"],mode="lines+markers",line=dict(color="blue"),name="Last Price",yaxis="y1")
fig2.update_layout(height=600,xaxis=dict(title="Time",rangeslider=dict(visible=True)),yaxis=dict(domain=[0.45,1],title="Last Price"),yaxis2=dict(domain=[0,0.35],title="Δ Turnover",type=axis_type),legend=dict(orientation="h"),hovermode="x unified")
st.plotly_chart(fig2,use_container_width=True)

# ============================================================
# 11. INTERVAL CLASS + SMA(ΔTT)
# ============================================================
def classify(row):
    s=np.sign(row["Δ totalTurnover"]) or 0
    c=np.sign(row["RollCorr"]) if not np.isnan(row["RollCorr"]) else 0
    sc=s*c
    return 1 if sc>0 else (-1 if sc<0 else 0)

turn_df["Signal_Val"]=turn_df.apply(classify,axis=1)
turn_df["Signal_Label"]=turn_df["Signal_Val"].map({1:"🟢 Bullish Turnover",0:"⚪ Neutral",-1:"🔴 Bearish Turnover"})
turn_df["SMA_ΔTT"]=turn_df["Δ totalTurnover"].rolling(5,min_periods=1).mean().round(2)
st.subheader("🧠 Turnover Behavior Insights (Per Interval)")
st.dataframe(turn_df[["time","Δ totalTurnover","SMA_ΔTT","Δ Price","RollCorr","Signal_Label"]])

# ============================================================
# 12. COMBINED SIGNALS
# ============================================================
st.subheader("🪄 Combined Signal Summary")
combo=turn_df[["time","RTR_Signal","TurnOsc_Signal","Signal_Label"]].copy()
st.dataframe(combo)

# ============================================================
# 13. CHART 4 – PRICE vs SMA(ΔTT)
# ============================================================
st.subheader("🆕 Chart 4 – Last Price (top) & SMA(Δ Turnover) (bottom)")
fig4=go.Figure()
fig4.add_trace(go.Bar(x=turn_df["time"],y=turn_df["SMA_ΔTT"],name="SMA Δ Turnover",marker_color="teal",opacity=0.6,yaxis="y2"))
fig4.add_trace(go.Scatter(x=turn_df["time"],y=turn_df["last_price"],mode="lines+markers",line=dict(color="blue"),name="Last Price",yaxis="y1"))
fig4.update_layout(height=600,xaxis=dict(title="Time",rangeslider=dict(visible=True)),yaxis=dict(domain=[0.45,1],title="Last Price"),yaxis2=dict(domain=[0,0.35],title="SMA Δ Turnover",type=axis_type),legend=dict(orientation="h"),hovermode="x unified")
st.plotly_chart(fig4,use_container_width=True)

# ============================================================
# 14. OVERALL DIRECTIONAL SIGNAL
# ============================================================
last_rows=turn_df.tail(5)
slope=np.sign(last_rows["Cum"].iloc[-1]-last_rows["Cum"].iloc[0])
corr_sign=np.sign(turn_df["RollCorr"].iloc[-1])
score=slope*corr_sign
if score>0:
    overall_signal="🟢 **Bullish Accumulation in Turnover**"
elif score<0:
    overall_signal="🔴 **Bearish Distribution in Turnover**"
else:
    overall_signal="⚪ **Neutral / Indecisive Turnover flow**"
st.subheader("📈 Overall Auto‑Signal")
st.markdown(overall_signal)
