import os
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from full_fred.fred import Fred

# ------------------------
# FRED 設定＆ヘルパー
# ------------------------
# ★ここで FRED_API_KEY を環境変数にセットしておくか、
#   事前にシェルや .env などで設定しておいてください。
#   例: export FRED_API_KEY="YOUR_FRED_API_KEY"
os.environ["FRED_API_KEY"] = "c716130f701440f2f42a576d781f767d"
fred = Fred()  # FRED_API_KEY を環境変数から読む

def get_fred_data(name, start="2013-01-01", end=""):
    """
    サンプルコードと同じ形式で FRED の時系列を取得する。
    index: date, column: 'value'
    """
    df = fred.get_series_df(name)[["date", "value"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.set_index("date")

    if end == "":
        df = df.loc[f"{start}":]
    else:
        df = df.loc[f"{start}":f"{end}"]
    return df

@st.cache_data
def load_fred_ohlcv(series_id: str, start="2013-01-01", end=""):
    """
    FRED の 'value' 時系列を、擬似 OHLCV に変換する。
    Open = High = Low = Close = value, Volume = 0
    として Index グラフ用に使う。
    """
    df_val = get_fred_data(series_id, start=start, end=end)
    if df_val is None or df_val.empty:
        return pd.DataFrame()

    if "value" not in df_val.columns:
        return pd.DataFrame()

    ohlcv = pd.DataFrame(index=df_val.index)
    ohlcv["Open"] = df_val["value"]
    ohlcv["High"] = df_val["value"]
    ohlcv["Low"] = df_val["value"]
    ohlcv["Close"] = df_val["value"]
    ohlcv["Volume"] = 0.0
    return ohlcv

# ------------------------
# Streamlit ページ設定
# ------------------------
st.set_page_config(page_title="Stock & Index Viewer with Interval-based MAs", layout="wide")

# ------------------------
# サイドバー（コントロール）
# ------------------------
st.sidebar.header("Stock Controls")

# デフォルトの Ticker（株）
default_tickers = ["VTI", "VXUS", "GLD", "VDE", "QYLD", "EWJ", "URTH"]

# デフォルトのインデックス Ticker (yfinance 側)
# ・^VIX：ボラティリティ指数
# ・EEM/EFA：EEM / EFA の比率を疑似インデックスとして表示
default_index_tickers = ["^VIX", "EEM/EFA", "1592.T", "^N500"]

# デフォルトのインデックス（FRED 系列）
# ここに "VIXCLS", "SP500" などの FRED シリーズIDを入れると
# 「Indexes」グラフにデフォルトで追加されます。
default_index_fred = ["SP500", "UNRATE", "T10Y2Y", "MEDCPIM158SFRBCLE", "DFEDTARU"]

# グラフの数 n（株価用）
n_charts = st.sidebar.number_input(
    "Number of stock tickers (n)",
    min_value=1,
    max_value=6,
    value=6,        # デフォルト
    step=1,
)

# 箱ヒゲ図（ローソク足）の期間（リサンプリング周期）
freq_map = {
    "1h": "1H",
    "1d": "1D",
    "1w": "1W",
    "1m": "1M",
}
interval_labels = list(freq_map.keys())
interval = st.sidebar.radio(
    "Candle interval (box period)",
    options=interval_labels,
    index=1,  # デフォルト: "1d"
)

# ★ Stocks / Indexes 共通の x軸期間設定（バー風 UI）
x_range_options = ["3m", "6m", "1y", "3y", "5y", "7y", "10y", "max"]
x_range_choice = st.sidebar.radio(
    "X-axis window (for 1d)",
    options=x_range_options,
    index=2,          # デフォルト: "1y"
    horizontal=True,  # バーっぽく横並びに
)

# Ticker の入力（株価用）
tickers = []
for i in range(n_charts):
    default_val = default_tickers[i] if i < len(default_tickers) else ""
    t = st.sidebar.text_input(f"Stock ticker symbol {i+1}", value=default_val)
    if t.strip():
        tickers.append(t.strip().upper())

if not tickers:
    st.warning("Stock ticker symbol を 1 つ以上入力してください。")
    st.stop()

# ------------------------
# インデックス用コントロール
# ------------------------
st.sidebar.header("Index Controls")

# ★ 期間指定は x_range_choice を流用するので
#    Index 用の別ラジオボタンは削除

n_index_extra = st.sidebar.number_input(
    "Number of additional index tickers",
    min_value=0,
    max_value=10,
    value=0,
    step=1,
)

index_extra_tickers = []
for i in range(n_index_extra):
    t = st.sidebar.text_input(f"Index ticker {i+1}", value="")
    if t.strip():
        index_extra_tickers.append(t.strip().upper())

# ------------------------
# データ読み込み（株価）
# ------------------------
@st.cache_data
def load_stock_df(ticker_symbol: str) -> pd.DataFrame:
    # yfinance から最大期間を取得
    df_raw = yf.download(tickers=ticker_symbol, period="max")
    if df_raw.empty:
        return df_raw

    # MultiIndex / 単一 Index 両対応で flatten
    def flatten_stock_df(df_multi: pd.DataFrame) -> pd.DataFrame:
        df = df_multi.copy()
        if isinstance(df.columns, pd.MultiIndex):
            # ("Close","AAPL") → "Close"
            df.columns = [col[0] for col in df.columns]
        df = df.reset_index().set_index("Date")
        df.index = pd.to_datetime(df.index)
        return df

    df_flat = flatten_stock_df(df_raw)

    # 必要な列だけ残す（存在するものだけ）
    cols_needed = ["Open", "High", "Low", "Close", "Volume"]
    cols_existing = [c for c in cols_needed if c in df_flat.columns]
    df_flat = df_flat[cols_existing]
    return df_flat

@st.cache_data
def load_stock_per(ticker_symbol: str):
    ticker = yf.Ticker(ticker_symbol)
    info = ticker.info

    # PER（Trailing P/E）を取得
    pe_ratio = info.get("trailingPE")   # 過去12か月の実績ベース
    # または前方予想P/Eを取得
    forward_pe = info.get("forwardPE")

    print("Trailing P/E:", pe_ratio)
    print("Forward P/E:", forward_pe)
    return pe_ratio

@st.cache_data
def load_ratio_df_eem_efa() -> pd.DataFrame:
    """
    EEM/EFA の比率から疑似 OHLCV を作成する。
    OHLC はそれぞれ EEM の OHLC / EFA の OHLC で定義。
    Volume は 0 で埋める。
    """
    df_eem = load_stock_df("EEM")
    df_efa = load_stock_df("EFA")
    if df_eem.empty or df_efa.empty:
        return pd.DataFrame()

    # 両方に共通する日付だけ
    idx = df_eem.index.intersection(df_efa.index)
    if idx.empty:
        return pd.DataFrame()

    df_eem = df_eem.loc[idx]
    df_efa = df_efa.loc[idx]

    ratio_df = pd.DataFrame(index=idx)
    for col in ["Open", "High", "Low", "Close"]:
        if col in df_eem.columns and col in df_efa.columns:
            ratio_df[col] = df_eem[col] / df_efa[col]
        else:
            # 必要な列が足りなければ空を返す
            return pd.DataFrame()

    # Volume はダミー（0）で作成
    ratio_df["Volume"] = 0.0
    return ratio_df

# ------------------------
# OHLCV を任意の頻度にまとめる関数
# ------------------------
def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    rule : '1H', '1D', '1W', '1M' などの resample ルール
    """
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    # Volume がない場合も動くように、共通列だけで agg を組み直す
    agg_used = {k: v for k, v in agg.items() if k in df.columns}
    df_resampled = df.resample(rule).agg(agg_used)
    # 全て NaN の期間は落とす（OHLC のどれかがあれば OK）
    cols_price = [c for c in ["Open", "High", "Low", "Close"] if c in df_resampled.columns]
    if cols_price:
        df_resampled = df_resampled.dropna(subset=cols_price)
    return df_resampled

# ------------------------
# interval に応じた 10/50/100/200 本分の移動平均
# ------------------------
def compute_ma_for_interval(df_resampled: pd.DataFrame) -> pd.DataFrame:
    """
    df_resampled : 既に '1H', '1D', '1W', '1M' などでリサンプリングされた DataFrame
    戻り値      : index は df_resampled.index、列は MA10, MA50, MA100, MA200
    """
    close = df_resampled["Close"]
    ma_df = pd.DataFrame(index=df_resampled.index)
    ma_df["MA10"] = close.rolling(window=10).mean()
    ma_df["MA50"] = close.rolling(window=50).mean()
    ma_df["MA100"] = close.rolling(window=100).mean()
    ma_df["MA200"] = close.rolling(window=200).mean()
    return ma_df

# ------------------------
# y軸レンジを「min→10%, max→90%」に配置するように計算
# ------------------------
def compute_axis_range(vmin: float, vmax: float):
    if pd.isna(vmin) or pd.isna(vmax):
        return None
    span = vmax - vmin
    if span <= 0:
        # すべて同じ値だったときは ±10% の余白
        if vmin == 0:
            return [0, 1]
        return [vmin * 0.9, vmax * 1.1]

    # min が 10%, max が 90% になるように軸レンジを設定
    axis_span = span / 0.8
    axis_min = vmin - 0.1 * axis_span  # = vmin - span/8
    axis_max = axis_min + axis_span
    return [axis_min, axis_max]

# Interval → 日数スケール（箱ヒゲの幅）
interval_to_days = {
    "1h": 1.0 / 24.0,
    "1d": 1.0,
    "1w": 7.0,
    "1m": 30.0,  # おおよその値
}

# Base x-axis range を日数換算（株価・インデックス共通）
def base_range_days(choice: str, full_start: pd.Timestamp, full_end: pd.Timestamp) -> float:
    """
    x_range_choice で選ばれた期間を「1日あたりの日数」に換算
    """
    if choice == "3m":
        return 30.0 * 3.0
    elif choice == "6m":
        return 30.0 * 6.0
    elif choice == "1y":
        return 365.0
    elif choice == "3y":
        return 365.0 * 3.0
    elif choice == "5y":
        return 365.0 * 5.0
    elif choice == "7y":
        return 365.0 * 7.0
    elif choice == "10y":
        return 365.0 * 10.0
    elif choice == "max":
        # 利用可能な全期間
        return max(1.0, (full_end - full_start).days)
    else:
        # デフォルトは 1年
        return 365.0

# interval → MA の単位ラベル（凡例用）
ma_unit_label = {
    "1h": "H",   # hours
    "1d": "D",   # days
    "1w": "W",   # weeks
    "1m": "M",   # months
}

# ------------------------
# グラフ生成関数（株価・インデックス共通）
# ------------------------
def build_figure(
    df_input: pd.DataFrame,
    label: str,
    per,
    base_range_choice: str,
    show_volume: bool = True,
    show_per_in_title: bool = True,
):
    # 選択された interval でリサンプリング
    rule = freq_map[interval]
    df_freq = resample_ohlcv(df_input, rule)
    if df_freq.empty:
        st.error(f"{label}: {interval} でリサンプリングできるデータがありません。")
        return None

    full_start = df_freq.index.min()
    full_end = df_freq.index.max()

    # 1d のときのベース期間（日数）を求める
    base_days = base_range_days(base_range_choice, full_start, full_end)

    # 箱ヒゲの期間に応じて、x軸の期間をスケーリング
    factor = interval_to_days.get(interval, 1.0) / interval_to_days["1d"]
    span_days = base_days * factor

    # 実際の表示期間
    start_candidate = full_end - pd.Timedelta(days=span_days)
    start = max(full_start, start_candidate)

    # 表示範囲（y軸レンジ計算用）
    df_view = df_freq.loc[(df_freq.index >= start) & (df_freq.index <= full_end)]
    if df_view.empty:
        df_view = df_freq.copy()
        start = df_view.index.min()

    # プロット用データ（全期間）
    df_plot = df_freq

    # y軸レンジ計算（Price）
    price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df_view.columns]
    price_min = df_view[price_cols].min().min()
    price_max = df_view[price_cols].max().max()
    price_range = compute_axis_range(price_min, price_max)

    # y軸レンジ計算（Volume）
    vol_range = None
    if show_volume and "Volume" in df_view.columns:
        vol_min = df_view["Volume"].min()
        vol_max = df_view["Volume"].max()
        vol_range = compute_axis_range(vol_min, vol_max)

    # interval に対応した 10/50/100/200 本分の移動平均を計算（全期間）
    ma_all = compute_ma_for_interval(df_freq)
    ma_for_plot = ma_all

    # 上昇/下落フラグ（Close >= Open → 上昇）全期間
    up_all = df_plot["Close"] >= df_plot["Open"]
    volume_colors_all = ["green" if is_up else "red" for is_up in up_all]

    # MA の凡例ラベル（単位を interval に合わせる）
    unit = ma_unit_label.get(interval, "")
    label_10 = f"MA 10{unit}" if unit else "MA 10"
    label_50 = f"MA 50{unit}" if unit else "MA 50"
    label_100 = f"MA 100{unit}" if unit else "MA 100"
    label_200 = f"MA 200{unit}" if unit else "MA 200"

    # Plotly 図の作成
    if show_volume:
        # 株価用：上段 Price、下段 Volume
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.8, 0.2],
        )
        row_price = 1
        row_volume = 2
    else:
        # インデックス用：Price のみ
        fig = make_subplots(
            rows=1,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.0,
            row_heights=[1.0],
        )
        row_price = 1
        row_volume = None

    # ---------------------------
    # 価格のプロット
    # ---------------------------
    if show_volume:
        # 株価用：ローソク足（箱ヒゲ）
        candle = go.Candlestick(
            x=df_plot.index,
            open=df_plot["Open"],
            high=df_plot["High"],
            low=df_plot["Low"],
            close=df_plot["Close"],
            name=f"Price ({interval})",
            increasing_line_color="green",
            increasing_fillcolor="green",
            decreasing_line_color="red",
            decreasing_fillcolor="red",
            showlegend=False,
        )
        fig.add_trace(candle, row=row_price, col=1)
    else:
        # インデックス用：Close の折れ線（#346FF4）
        price_line = go.Scatter(
            x=df_plot.index,
            y=df_plot["Close"],
            mode="lines",
            name=f"Price ({interval})",
            line=dict(color="#346FF4", width=1.5),
        )
        fig.add_trace(price_line, row=row_price, col=1)

    # ---------------------------
    # 出来高（株価のみ）
    # ---------------------------
    if show_volume and "Volume" in df_plot.columns and row_volume is not None:
        volume = go.Bar(
            x=df_plot.index,
            y=df_plot["Volume"],
            name=f"Volume ({interval})",
            marker=dict(color=volume_colors_all),
            showlegend=False,
        )
        fig.add_trace(volume, row=row_volume, col=1)

    # ---------------------------
    # 移動平均線
    # ---------------------------
    if show_volume:
        # 株価用：10/50/100 本分移動平均線（全期間）
        fig.add_trace(
            go.Scatter(
                x=ma_for_plot.index,
                y=ma_for_plot["MA10"],
                mode="lines",
                name=label_10,
                line=dict(width=1.5),
            ),
            row=row_price,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=ma_for_plot.index,
                y=ma_for_plot["MA50"],
                mode="lines",
                name=label_50,
                line=dict(width=1.5, dash="dash"),
            ),
            row=row_price,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=ma_for_plot.index,
                y=ma_for_plot["MA100"],
                mode="lines",
                name=label_100,
                line=dict(width=1.5, dash="dot"),
            ),
            row=row_price,
            col=1,
        )
    else:
        # インデックス用：200 本分移動平均線のみ（黒線）
        fig.add_trace(
            go.Scatter(
                x=ma_for_plot.index,
                y=ma_for_plot["MA200"],
                mode="lines",
                name=label_200,
                line=dict(color="black", width=1.5),
            ),
            row=row_price,
            col=1,
        )

    # レイアウト設定
    if show_volume:
        fig.update_layout(
            dragmode="pan",
            showlegend=True,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=0, r=0, t=0, b=0),
            height=300,
        )
    else:
        fig.update_layout(
            dragmode="pan",
            showlegend=True,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=0, r=0, t=0, b=0),
            height=250,
        )

    # x軸レンジ（初期表示だけ制限、データ自体は全期間）
    fig.update_xaxes(
        range=[start, full_end],
        rangeslider_visible=False,
        row=row_price,
        col=1,
    )
    if show_volume and row_volume is not None:
        fig.update_xaxes(
            range=[start, full_end],
            rangeslider_visible=False,
            row=row_volume,
            col=1,
        )

    # タイトル文字列
    if show_per_in_title:
        if per is not None:
            title_text = f"{label} PER:{per:.1f}"
        else:
            title_text = f"{label} PER:NA"
    else:
        # インデックス用：シンボルのみ表示
        title_text = label

    fig.add_annotation(
        x=0.5,
        y=1.0,
        xref="paper",
        yref="paper",
        text=title_text,
        showarrow=False,
        font=dict(size=16),
    )

    # y軸レンジ（Price / Volume）
    if price_range is not None:
        fig.update_yaxes(range=price_range, row=row_price, col=1)
    if show_volume and vol_range is not None and row_volume is not None:
        fig.update_yaxes(range=vol_range, row=row_volume, col=1)

    return fig

# ------------------------
# 上側：株価グリッド表示
# ------------------------
st.markdown("## Stocks")

n_cols = 3
chart_idx = 0
cols = None

for ticker in tickers:
    stock_df = load_stock_df(ticker)
    per = load_stock_per(ticker)

    if stock_df.empty:
        st.error(f"{ticker}: データが取得できませんでした。Ticker シンボルを確認してください。")
        continue

    fig = build_figure(
        stock_df,
        ticker,
        per,
        base_range_choice=x_range_choice,  # ★共通期間設定を使用
        show_volume=True,
        show_per_in_title=True,   # 株価：PER をタイトルに表示
    )
    if fig is None:
        continue

    if chart_idx % n_cols == 0:
        cols = st.columns(n_cols)

    col = cols[chart_idx % n_cols]
    with col:
        st.plotly_chart(fig, use_container_width=True)

    chart_idx += 1

# ------------------------
# 下側：インデックスグラフ（値のみ）
# ------------------------
st.markdown("## Indexes")

# デフォルトインデックス + 追加インデックス (yfinance 側)
index_tickers = default_index_tickers + index_extra_tickers

n_cols_idx = 3
idx_chart = 0
cols_idx = None

# まず yfinance のインデックス
for idx_ticker in index_tickers:
    # EEM/EFA は疑似インデックス（比率）
    if idx_ticker == "EEM/EFA":
        df_idx = load_ratio_df_eem_efa()
        if df_idx.empty:
            st.warning("EEM/EFA のインデックスデータが取得できませんでした。")
            continue
    else:
        df_idx = load_stock_df(idx_ticker)
        if df_idx.empty:
            st.error(f"{idx_ticker}: インデックスデータが取得できませんでした。Ticker シンボルを確認してください。")
            continue

    fig_idx = build_figure(
        df_idx,
        idx_ticker,
        per=None,
        base_range_choice=x_range_choice,  # ★Stocks と同じ期間設定を使用
        show_volume=False,          # インデックス：出来高を表示しない（＝line表示）
        show_per_in_title=False,    # インデックス：タイトルはシンボルのみ
    )
    if fig_idx is None:
        continue

    if idx_chart % n_cols_idx == 0:
        cols_idx = st.columns(n_cols_idx)

    col_idx = cols_idx[idx_chart % n_cols_idx]
    with col_idx:
        st.plotly_chart(fig_idx, use_container_width=True)

    idx_chart += 1

# 次に FRED 系列のインデックスを追加
for fred_series in default_index_fred:
    df_fred = load_fred_ohlcv(fred_series)
    if df_fred.empty:
        st.warning(f"{fred_series}: FRED データが取得できませんでした。FRED シリーズIDや API キーを確認してください。")
        continue

    fig_fred = build_figure(
        df_fred,
        fred_series,
        per=None,
        base_range_choice=x_range_choice,  # ★共通期間設定
        show_volume=False,          # インデックス：出来高なし
        show_per_in_title=False,    # タイトルはシリーズIDのみ
    )
    if fig_fred is None:
        continue

    if idx_chart % n_cols_idx == 0:
        cols_idx = st.columns(n_cols_idx)

    col_idx = cols_idx[idx_chart % n_cols_idx]
    with col_idx:
        st.plotly_chart(fig_fred, use_container_width=True)

    idx_chart += 1
