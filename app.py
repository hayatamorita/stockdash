import os
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import datetime as dt
from plotly.subplots import make_subplots
from full_fred.fred import Fred

# ============================================================
# 0. 定数・グローバル設定
# ============================================================

# ---- FRED 設定 ----
FRED_API_KEY = "c716130f701440f2f42a576d781f767d"
os.environ["FRED_API_KEY"] = FRED_API_KEY
fred = Fred()  # FRED_API_KEY を環境変数から読む

# ---- デフォルト Ticker 設定 ----
DEFAULT_STOCK_TICKERS = ["VTI", "VXUS", "QYLD", "URTH", "VDE", "VDC", "CPER", "GLD"]
DEFAULT_INDEX_TICKERS_YF = ["^VIX", "EEM/EFA", "^IXIC", "^DJI", "1592.T", "^N500"]
DEFAULT_INDEX_TICKERS_FRED = ["SP500", "DEXJPUS", "DGS10", "FEDFUNDS", "UNRATE", "MEDCPIM158SFRBCLE", "GDPC1", "T10Y2Y", "DFEDTARU"]

# ---- インデックス説明テキスト ----
INDEX_DESCRIPTION_MAP = {
    "1592.T": "TOPIX",
    "^VIX": "VIX",
    "^N500": "Nikkei 500",
    "UNRATE": "Unemployment Rate",
    "T10Y2Y": "10Y-2Y Treasury",
    "MEDCPIM158SFRBCLE": "Median CPI",
    "DFEDTARU": "Fed Funds Upper Target",
    "^IXIC": "Nasdaq",
    "^DJI": "Dow",
    "CPER": "Copper",
    "^GSPC": "SP500",
    "DEXJPUS": "USD/JPY",
    "FEDFUNDS": "Federal Funds Rate",
    "SP500": "SP500",
    "DGS10": "10-year Treasury yield",
    "GDPC1": "GDP YOY",
    "EEM/EFA": "EEM/EFA",
}

# ---- ローソク足のリサンプリング周期 ----
FREQ_MAP = {
    "1h": "1H",
    "1d": "1D",
    "1w": "1W",
    "1m": "1M",
}

# ---- interval → 日数スケール（箱ヒゲの幅用） ----
INTERVAL_TO_DAYS = {
    "1h": 1.0 / 24.0,
    "1d": 1.0,
    "1w": 7.0,
    "1m": 30.0,  # おおよその値
}

# ---- interval → MA の単位ラベル（凡例用） ----
MA_UNIT_LABEL = {
    "1h": "H",   # hours
    "1d": "D",   # days
    "1w": "W",   # weeks
    "1m": "M",   # months
}

# ---- YTD 棒グラフ用 ----
YTD_TICKERS = ["EPOL", "VNM", "EWW", "MCHI", "ECH", "EWZ", "EWG", "VXUS", "SPY", "EPI", "EWJ"]
COUNTRY_MAP_YTD = {
    "EPOL": "POL",   # Poland
    "VNM": "VNM",    # Vietnam
    "EWW": "MEX",    # Mexico
    "MCHI": "CHN",   # China
    "ECH": "CHL",    # Chile
    "EWZ": "BRA",    # Brazil
    "EWG": "GER",    # Germany
    "VXUS": "INTL",  # International ex-US
    "SPY": "USA",    # United States
    "EPI": "IND",    # India
    "EWJ": "JPN",    # Japan
}
YEAR_START_YTD = "2025-01-01"

# ============================================================
# 1. FRED 系ヘルパー
# ============================================================

def get_fred_data(name: str, start: str = "2013-01-01", end: str = "") -> pd.DataFrame:
    """
    サンプルコードと同じ形式で FRED の時系列を取得する。
    index: date, column: 'value'
    """
    df = fred.get_series_df(name)[["date", "value"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.set_index("date")

    if end == "":
        return df.loc[f"{start}":]
    else:
        return df.loc[f"{start}":f"{end}"]


@st.cache_data
def load_fred_ohlcv(series_id: str, start: str = "2013-01-01", end: str = "") -> pd.DataFrame:
    """
    FRED の 'value' 時系列を、擬似 OHLCV に変換する。
    Open = High = Low = Close = value, Volume = 0
    としてインデックスグラフ用に使う。
    """
    df_val = get_fred_data(series_id, start=start, end=end)
    if df_val is None or df_val.empty or "value" not in df_val.columns:
        return pd.DataFrame()

    ohlcv = pd.DataFrame(index=df_val.index)
    ohlcv["Open"] = df_val["value"]
    ohlcv["High"] = df_val["value"]
    ohlcv["Low"] = df_val["value"]
    ohlcv["Close"] = df_val["value"]
    ohlcv["Volume"] = 0.0
    return ohlcv

def calc_gdp_yoy_df(df):
    """
    GDPC1のOHLC DataFrameを入力し、
    前年同期比 YoY (%) を元dfと同じ列構造で返す。

    Parameters
    ----------
    df : pd.DataFrame
        columns: ['Open', 'High', 'Low', 'Close', 'Volume']

    Returns
    -------
    df_yoy : pd.DataFrame
        同じ列構造で YoY をセットした DataFrame
    """
    # GDP値は Close から取る
    gdp = df['Close']

    # YoY 計算
    gdp_yoy = (gdp / gdp.shift(4) - 1) * 100

    # 元の列構造を引き継ぐ
    df_yoy = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)

    # OHLC に同じ値を入れる
    df_yoy['Open']  = gdp_yoy
    df_yoy['High']  = gdp_yoy
    df_yoy['Low']   = gdp_yoy
    df_yoy['Close'] = gdp_yoy
    
    # Volume は 0 にする or NaN にしたければ df_yoy['Volume'] = np.nan
    df_yoy['Volume'] = 0.0

    return df_yoy

# ============================================================
# 2. yfinance データ取得ヘルパー
# ============================================================

def _flatten_stock_df(df_multi: pd.DataFrame) -> pd.DataFrame:
    """MultiIndex / 単一 Index 両対応で列をフラット化。"""
    df = df_multi.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # ("Close", "AAPL") → "Close"
        df.columns = [col[0] for col in df.columns]
    df = df.reset_index().set_index("Date")
    df.index = pd.to_datetime(df.index)
    return df


@st.cache_data
def load_stock_df(ticker_symbol: str) -> pd.DataFrame:
    """
    yfinance から最大期間を取得し、OHLCV に整形する。
    """
    df_raw = yf.download(tickers=ticker_symbol, period="max", auto_adjust=False)
    if df_raw.empty:
        return df_raw

    df_flat = _flatten_stock_df(df_raw)
    cols_needed = ["Open", "High", "Low", "Close", "Volume"]
    cols_existing = [c for c in cols_needed if c in df_flat.columns]
    return df_flat[cols_existing]


@st.cache_data
def load_stock_per(ticker_symbol: str):
    """
    PER（Trailing P/E）を取得。
    """
    ticker = yf.Ticker(ticker_symbol)
    info = ticker.info
    pe_ratio = info.get("trailingPE")   # 過去12か月の実績ベース
    forward_pe = info.get("forwardPE")  # 未使用だが残しておく（将来拡張用）
    print("pe_ratio:", pe_ratio)
    return pe_ratio


@st.cache_data
def load_stock_eps(ticker_symbol: str):
    """
    EPS（Trailing EPS）を取得。
    """
    ticker = yf.Ticker(ticker_symbol)
    info = ticker.info
    eps_trailing = info.get("trailingEps")
    eps_forward = info.get("forwardEps")  # 未使用だが残しておく（将来拡張用）
    print(ticker_symbol)
    print("eps:", eps_trailing)
    print("for eps:", eps_forward)
    return eps_trailing


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
            return pd.DataFrame()  # 必要列が足りなければ空
    ratio_df["Volume"] = 0.0
    return ratio_df

# ============================================================
# 3. 計算ヘルパー（リサンプリング・MA・レンジ）
# ============================================================

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    OHLCV を任意の頻度にまとめる。
    rule : '1H', '1D', '1W', '1M' などの resample ルール
    """
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    agg_used = {k: v for k, v in agg.items() if k in df.columns}
    df_resampled = df.resample(rule).agg(agg_used)

    cols_price = [c for c in ["Open", "High", "Low", "Close"] if c in df_resampled.columns]
    if cols_price:
        df_resampled = df_resampled.dropna(subset=cols_price)
    return df_resampled


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


def compute_axis_range(vmin: float, vmax: float):
    """
    y軸レンジを「min→10%, max→90%」に配置するように計算。
    """
    if pd.isna(vmin) or pd.isna(vmax):
        return None

    span = vmax - vmin
    if span <= 0:
        if vmin == 0:
            return [0, 1]
        return [vmin * 0.9, vmax * 1.1]

    axis_span = span / 0.8
    axis_min = vmin - 0.1 * axis_span
    axis_max = axis_min + axis_span
    return [axis_min, axis_max]


def base_range_days(choice: str, full_start: pd.Timestamp, full_end: pd.Timestamp) -> float:
    """
    x_range_choice で選ばれた期間を「日数」に換算。
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
        return max(1.0, (full_end - full_start).days)
    else:
        return 365.0  # デフォルト 1年

# ============================================================
# 4. Plotly 図生成関数
# ============================================================

def _calc_view_window(
    df_freq: pd.DataFrame,
    interval: str,
    base_range_choice: str,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]:
    """
    interval と base_range_choice に応じて、表示期間を計算。
    戻り値: (start, full_end, df_view)
    """
    full_start = df_freq.index.min()
    full_end = df_freq.index.max()

    base_days = base_range_days(base_range_choice, full_start, full_end)

    factor = INTERVAL_TO_DAYS.get(interval, 1.0) / INTERVAL_TO_DAYS["1d"]
    span_days = base_days * factor

    start_candidate = full_end - pd.Timedelta(days=span_days)
    start = max(full_start, start_candidate)

    df_view = df_freq.loc[(df_freq.index >= start) & (df_freq.index <= full_end)]
    if df_view.empty:
        df_view = df_freq.copy()
        start = df_view.index.min()

    return start, full_end, df_view


def _make_title(
    label: str,
    per,
    eps,
    show_per_in_title: bool,
    index_desc: str = "",
) -> str:
    """
    株価・インデックス共通のタイトル文字列を生成。
    """
    if show_per_in_title:
        if per is not None and eps is not None:
            return f"{label} PER:{per:.1f} EPS:{eps:.1f}"
        elif per is not None and eps is None:
            return f"{label} PER:{per:.1f}"
        else:
            return f"{label} PER:NA"
    else:
        if index_desc:
            #return f"{label}:{index_desc}"
            return f"{index_desc}"
        return label


def build_figure(
    df_input: pd.DataFrame,
    label: str,
    per,
    eps,
    interval: str,
    base_range_choice: str,
    show_volume: bool = True,
    show_per_in_title: bool = True,
    index_desc: str = "",
):
    """
    株価・インデックス共通のグラフ生成関数。
    interval でリサンプリングし、OHLC や MA、Volume を描画する。
    """
    if interval not in FREQ_MAP:
        st.error(f"Unknown interval: {interval}")
        return None

    # ---- リサンプリング ----
    rule = FREQ_MAP[interval]
    df_freq = resample_ohlcv(df_input, rule)
    if df_freq.empty:
        st.error(f"{label}: {interval} でリサンプリングできるデータがありません。")
        return None

    start, full_end, df_view = _calc_view_window(df_freq, interval, base_range_choice)
    df_plot = df_freq

    # ---- y軸レンジ（Price / Volume） ----
    price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df_view.columns]
    price_min = df_view[price_cols].min().min()
    price_max = df_view[price_cols].max().max()
    price_range = compute_axis_range(price_min, price_max)

    vol_range = None
    if show_volume and "Volume" in df_view.columns:
        vol_min = df_view["Volume"].min()
        vol_max = df_view["Volume"].max()
        vol_range = compute_axis_range(vol_min, vol_max)

    # ---- MA 計算 ----
    ma_all = compute_ma_for_interval(df_freq)
    ma_for_plot = ma_all

    # ---- 上昇/下落フラグ ----
    up_all = df_plot["Close"] >= df_plot["Open"]
    volume_colors_all = ["green" if is_up else "red" for is_up in up_all]

    # ---- MA 凡例ラベル ----
    unit = MA_UNIT_LABEL.get(interval, "")
    label_10 = f"MA 10{unit}" if unit else "MA 10"
    label_50 = f"MA 50{unit}" if unit else "MA 50"
    label_100 = f"MA 100{unit}" if unit else "MA 100"
    label_200 = f"MA 200{unit}" if unit else "MA 200"

    # ---- Figure 準備 ----
    if show_volume:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.0,
            row_heights=[0.8, 0.2],
        )
        row_price = 1
        row_volume = 2
    else:
        fig = make_subplots(
            rows=1,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.0,
            row_heights=[1.0],
        )
        row_price = 1
        row_volume = None

    # ---- 価格のプロット ----
    if show_volume:
        # 株価用：ローソク足
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
        # インデックス用：Close の折れ線
        price_line = go.Scatter(
            x=df_plot.index,
            y=df_plot["Close"],
            mode="lines",
            name=f"Price ({interval})",
            line=dict(color="#346FF4", width=1.5),
        )
        fig.add_trace(price_line, row=row_price, col=1)

    # ---- 出来高（株価のみ） ----
    if show_volume and "Volume" in df_plot.columns and row_volume is not None:
        volume = go.Bar(
            x=df_plot.index,
            y=df_plot["Volume"],
            name=f"Volume ({interval})",
            marker=dict(color=volume_colors_all),
            showlegend=False,
        )
        fig.add_trace(volume, row=row_volume, col=1)

    # ---- 移動平均線 ----
    if show_volume:
        # 株価用：10/50/100 本分
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
        # インデックス用：200 本分
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

    # ---- タイトル ----
    title_text = _make_title(label, per, eps, show_per_in_title, index_desc=index_desc)
    fig.add_annotation(
        x=0.5,
        y=1.0,
        xref="paper",
        yref="paper",
        text=title_text,
        showarrow=False,
        font=dict(size=16),
    )

    # ---- レイアウト共通設定 ----
    layout_common = dict(
        dragmode="pan",
        showlegend=True,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=0, b=0),
    )
    if show_volume:
        layout_common["height"] = 250
    else:
        layout_common["height"] = 200
    fig.update_layout(**layout_common)

    # ---- x軸レンジ（初期表示） ----
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

    # ---- y軸レンジ ----
    if price_range is not None:
        fig.update_yaxes(range=price_range, row=row_price, col=1)
    if show_volume and row_volume is not None and vol_range is not None:
        fig.update_yaxes(range=vol_range, row=row_volume, col=1)

    return fig

# ============================================================
# 5. UI 用コンポーネント関数（図を返すだけにする）
# ============================================================

def get_stock_figures(
    tickers: list[str],
    interval: str,
    x_range_choice: str,
):
    """株価セクションで表示する Figure のリストを返す。"""
    figs = []

    for ticker in tickers:
        stock_df = load_stock_df(ticker)
        per = load_stock_per(ticker)
        eps = load_stock_eps(ticker)

        if stock_df.empty:
            st.error(f"{ticker}: データが取得できませんでした。Ticker シンボルを確認してください。")
            continue

        fig = build_figure(
            stock_df,
            label=ticker,
            per=per,
            eps=eps,
            interval=interval,
            base_range_choice=x_range_choice,
            show_volume=True,
            show_per_in_title=True,
        )
        if fig is not None:
            figs.append(fig)

    return figs


def get_index_figures(
    interval: str,
    x_range_choice: str,
    default_index_tickers: list[str],
    extra_index_tickers: list[str],
    fred_series_list: list[str],
):
    """インデックスセクションで表示する Figure のリストを返す。"""
    figs = []
    # ---- FRED 系列インデックス ----
    for fred_series in fred_series_list:
        df_fred = load_fred_ohlcv(fred_series)
        if fred_series == 'GDPC1':
            df_fred = calc_gdp_yoy_df(df_fred)
        if df_fred.empty:
            st.warning(f"{fred_series}: FRED データが取得できませんでした。FRED シリーズIDや API キーを確認してください。")
            continue

        fred_desc = INDEX_DESCRIPTION_MAP.get(fred_series, "")

        fig_fred = build_figure(
            df_fred,
            label=fred_series,
            per=None,
            eps=None,
            interval=interval,
            base_range_choice=x_range_choice,
            show_volume=False,
            show_per_in_title=False,
            index_desc=fred_desc,
        )
        if fig_fred is not None:
            figs.append(fig_fred)

    # ---- yfinance インデックス ----
    index_tickers = default_index_tickers + extra_index_tickers

    for idx_ticker in index_tickers:
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

        idx_desc = INDEX_DESCRIPTION_MAP.get(idx_ticker, "")

        fig_idx = build_figure(
            df_idx,
            label=idx_ticker,
            per=None,
            eps=None,
            interval=interval,
            base_range_choice=x_range_choice,
            show_volume=False,
            show_per_in_title=False,
            index_desc=idx_desc,
        )
        if fig_idx is not None:
            figs.append(fig_idx)


    return figs


def build_ytd_figure():
    """
    YTD パフォーマンスの棒グラフ Figure を返す。
    ストリームリットのレイアウト(st.columns, with ...) は main() 側で行う。
    """
    ytd_results = []
    for ticker in YTD_TICKERS:
        data = yf.download(ticker, start=YEAR_START_YTD, auto_adjust=False)
        if len(data) < 2:
            st.warning(f"{ticker} のYTD計算に必要なデータが不足しています")
            continue

        open_price = data.iloc[0]["Open"]
        current_price = data.iloc[-1]["Close"]
        ytd_val = float(((current_price / open_price - 1) * 100).iloc[0])

        ytd_results.append(
            {
                "Ticker": ticker,
                "YTD": ytd_val,
                "Country": COUNTRY_MAP_YTD[ticker],
            }
        )

    if not ytd_results:
        st.info("その他のYTDパフォーマンスを計算できるデータがありませんでした。")
        return None

    df_ytd = pd.DataFrame(ytd_results).sort_values("YTD", ascending=False)

    # x軸ラベル（Ticker + 国名略称）
    x_labels_ytd = [f"{t}<br>{c}" for t, c in zip(df_ytd["Ticker"], df_ytd["Country"])]

    # SPY だけ赤
    colors_ytd = ["red" if t == "SPY" else "#1f567d" for t in df_ytd["Ticker"]]

    fig_ytd = go.Figure(
        data=[
            go.Bar(
                x=x_labels_ytd,
                y=df_ytd["YTD"],
                marker_color=colors_ytd,
                text=[f"{y:.2f}%" for y in df_ytd["YTD"]],
                textposition="outside",
            )
        ]
    )

    fig_ytd.update_layout(
        title="Year-to-Date Performance (YTD)",
        yaxis_title="YTD (%)",
        template="plotly_white",
        xaxis_tickfont=dict(size=12),
        margin=dict(l=40, r=40, t=80, b=50),
        height=400,
    )

    return fig_ytd


def build_ytd_year_figure() -> go.Figure | None:
    # -----------------------------
    # 2. 対象となる過去10年間（フルイヤー）＋ 今年(YTD)
    #    例：今が2025年なら 2015〜2024年がフルイヤー,
    #        2025年は年初〜今日までのYTD
    # -----------------------------
    today = dt.date.today()
    current_year = today.year

    start_year = current_year - 10        # 10年前の年（フルイヤーの最初）
    end_year = current_year - 1           # 直近のフルイヤーの最後
    years_full = list(range(start_year, end_year + 1))

    records = []

    # -----------------------------
    # 3-1. フルイヤーの年次リターン計算（〜昨年まで）
    # -----------------------------
    for ticker in YTD_TICKERS:
        for year in years_full:
            year_start = f"{year}-01-01"
            year_end   = f"{year + 1}-01-01"  # 翌年の1/1まで取得して、その直前が年末

            data = yf.download(ticker, start=year_start, end=year_end, progress=False, auto_adjust=False)

            # データ不足の場合はスキップ
            if len(data) < 2:
                print(f"{ticker} の {year} 年のデータが不足しています")
                continue

            open_price = data.iloc[0]["Open"]     # 年初の営業日の始値
            close_price = data.iloc[-1]["Close"]  # 年末の営業日の終値
            annual_return = float(((close_price / open_price - 1) * 100).iloc[0])

            records.append({
                "Year": year,
                "Ticker": ticker,
                "Country": COUNTRY_MAP_YTD[ticker],
                "Return(%)": annual_return
            })

    # -----------------------------
    # 3-2. 今年のYTDリターン計算
    #      年初(1/1以降の最初の営業日)〜「現在まで」の成長率
    # -----------------------------
    ytd_start = dt.date(current_year, 1, 1).isoformat()
    # yfinance の end は「その前日」まで取得なので、今日の翌日を指定
    ytd_end = (today + dt.timedelta(days=1)).isoformat()

    for ticker in YTD_TICKERS:
        data = yf.download(ticker, start=ytd_start, end=ytd_end, progress=False, auto_adjust=False)

        if len(data) < 2:
            print(f"{ticker} の {current_year} 年(YTD)のデータが不足しています")
            continue

        open_price = data.iloc[0]["Open"]     # 今年最初の営業日の始値
        close_price = data.iloc[-1]["Close"]  # 現在時点での直近営業日の終値
        #ytd_return = float((close_price / open_price - 1) * 100)
        ytd_return = float(((close_price / open_price - 1) * 100).iloc[0])

        records.append({
            "Year": current_year,
            "Ticker": ticker,
            "Country": COUNTRY_MAP_YTD[ticker],
            "Return(%)": ytd_return
        })

    # -----------------------------
    # 4. 長い形式の DataFrame & 国ごとのピボット
    # -----------------------------
    df_long = pd.DataFrame(records)

    # 国ごとの成長率テーブル（行：Year, 列：Country）
    df_country = df_long.pivot_table(
        index="Year",
        columns="Country",
        values="Return(%)"
    ).sort_index()

    print("=== 国ごとの年次リターン DataFrame ===")
    print(df_country)

    # -----------------------------
    # 5. Plotly ドット＋線グラフ作成
    #    （各国ごとに lines+markers で比較）
    # -----------------------------
    fig = go.Figure()

    # 国ごとに線を引く
    for country in sorted(df_long["Country"].unique()):
        country_data = df_long[df_long["Country"] == country].sort_values("Year")

        # SPY(USA)だけ色を変えたい場合
        if country == "USA":
            line_color = "red"
        else:
            line_color = None  # Plotly におまかせ

        fig.add_trace(
            go.Scatter(
                x=country_data["Year"],
                y=country_data["Return(%)"],
                mode="lines+markers",          # ドット＋線
                name=country,
                line=dict(color=line_color) if line_color else None
            )
        )

    fig.update_layout(
        title=f"Annual Returns by Country ({start_year}–{current_year}) ",
        xaxis_title="Year",
        yaxis_title="Return (%)",
        template="plotly_white",
        hovermode="x unified",
        legend_title="Country",
        margin=dict(l=40, r=40, t=80, b=40),
    )

    return fig



# ============================================================
# 6. メインアプリ（Streamlit の構成はここに集約）
# ============================================================

def main():
    # ------------------------
    # Streamlit ページ設定
    # ------------------------
    st.set_page_config(
        page_title="StockDash",
        layout="wide",
    )

    st.sidebar.title("📈StockDash")

    # ------------------------
    # サイドバー（株コントロール）
    # ------------------------
    st.sidebar.header("Stock Controls")

    # グラフの数 n（株価用）
    n_charts = st.sidebar.number_input(
        "Number of stock tickers (n)",
        min_value=1,
        value=len(DEFAULT_STOCK_TICKERS),
        step=1,
    )

    # interval 選択
    interval_labels = list(FREQ_MAP.keys())
    interval = st.sidebar.radio(
        "Candle interval (box period)",
        options=interval_labels,
        index=1,  # デフォルト: "1d"
    )

    # x軸期間選択
    x_range_options = ["3m", "6m", "1y", "3y", "5y", "7y", "10y", "max"]
    x_range_choice = st.sidebar.radio(
        "X-axis window (for 1d)",
        options=x_range_options,
        index=2,  # デフォルト: "1y"
        horizontal=True,
    )

    # Ticker 入力
    tickers: list[str] = []
    for i in range(n_charts):
        default_val = DEFAULT_STOCK_TICKERS[i] if i < len(DEFAULT_STOCK_TICKERS) else ""
        t = st.sidebar.text_input(f"Stock ticker symbol {i+1}", value=default_val)
        if t.strip():
            tickers.append(t.strip().upper())

    if not tickers:
        st.warning("Stock ticker symbol を 1 つ以上入力してください。")
        st.stop()

    # ------------------------
    # サイドバー（インデックスコントロール）
    # ------------------------
    st.sidebar.header("Index Controls")

    n_index_extra = st.sidebar.number_input(
        "Number of additional index tickers",
        min_value=0,
        max_value=10,
        value=0,
        step=1,
    )

    index_extra_tickers: list[str] = []
    for i in range(n_index_extra):
        t = st.sidebar.text_input(f"Index ticker {i+1}", value="")
        if t.strip():
            index_extra_tickers.append(t.strip().upper())

    # ------------------------
    # メインレイアウト構成（st.columns / with は全てここ）
    # ------------------------
    n_cols = 4

    # ===== Stocks セクション =====
    st.markdown("## Stocks")
    stock_figs = get_stock_figures(
        tickers=tickers,
        interval=interval,
        x_range_choice=x_range_choice,
    )

    if stock_figs:
        for idx, fig in enumerate(stock_figs):
            if idx % n_cols == 0:
                cols = st.columns(n_cols)
            col = cols[idx % n_cols]
            with col:
                st.plotly_chart(fig, width='content')

    # ===== Indexes セクション =====
    st.markdown("## Indexes")
    index_figs = get_index_figures(
        interval=interval,
        x_range_choice=x_range_choice,
        default_index_tickers=DEFAULT_INDEX_TICKERS_YF,
        extra_index_tickers=index_extra_tickers,
        fred_series_list=DEFAULT_INDEX_TICKERS_FRED,
    )

    if index_figs:
        for idx, fig in enumerate(index_figs):
            if idx % n_cols == 0:
                cols = st.columns(n_cols)
            col = cols[idx % n_cols]
            with col:
                st.plotly_chart(fig, width='content')

    # ===== その他（YTD）セクション =====
    st.markdown("## Others")
    fig_ytd = build_ytd_figure()
    fig_ytd_year = build_ytd_year_figure()
    if fig_ytd is not None:
        # misc_cols = st.columns(3)
        # with misc_cols[0]:
        #     st.plotly_chart(fig_ytd, width='content')
        col_left, col_right = st.columns([1, 2])
        with col_left:
            st.plotly_chart(fig_ytd, width='content')
        with col_right:
            st.plotly_chart(fig_ytd_year, width='content')


if __name__ == "__main__":
    main()
