import os
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import datetime as dt
from plotly.subplots import make_subplots
from full_fred.fred import Fred
from concurrent.futures import ThreadPoolExecutor, as_completed

# =====================================
#  FRED 設定 & 定数
# =====================================
os.environ["FRED_API_KEY"] = "c716130f701440f2f42a576d781f767d"
fred = Fred()  # FRED_API_KEY を環境変数から読む

# --- デフォルト設定 ---
DEFAULT_STOCK_TICKERS = ["VTI", "VXUS", "QYLD", "URTH", "VDE", "VDC", "CPER", "GLD"]
DEFAULT_INDEX_TICKERS_YF = ["^VIX", "EEM/EFA", "^GSPC", "^IXIC", "^DJI", "1592.T", "^N500"]
DEFAULT_INDEX_FRED = ["UNRATE", "T10Y2Y", "MEDCPIM158SFRBCLE", "DFEDTARU"]

# YTD 用
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
YTD_START = "2025-01-01"

# interval → pandas resample rule
FREQ_MAP = {
    "1h": "1H",
    "1d": "1D",
    "1w": "1W",
    "1m": "1M",
}

# interval → 1本あたりのおおよその日数
INTERVAL_TO_DAYS = {
    "1h": 1.0 / 24.0,
    "1d": 1.0,
    "1w": 7.0,
    "1m": 30.0,  # おおよその値
}

# interval → MA 単位ラベル
MA_UNIT_LABEL = {
    "1h": "H",   # hours
    "1d": "D",   # days
    "1w": "W",   # weeks
    "1m": "M",   # months
}

# x軸範囲オプション
X_RANGE_OPTIONS = ["3m", "6m", "1y", "3y", "5y", "7y", "10y", "max"]

# インデックス説明テキスト
INDEX_DESCRIPTION_MAP = {
    "1592.T": "TOPIX",
    "^VIX": "VIX Volatility Index",
    "^N500": "Nikkei 500",
    "UNRATE": "Unemployment Rate",
    "T10Y2Y": "10Y-2Y Treasury",
    "MEDCPIM158SFRBCLE": "Median CPI",
    "DFEDTARU": "Fed Funds Upper Target",
    "^IXIC": "Nasdaq",
    "^DJI": "Dow",
    "CPER": "Copper",
    "^GSPC": "SP500"
    # 必要に応じて追加
}


# =====================================
#  FRED 関連
# =====================================
def get_fred_data(name: str, start: str = "2013-01-01", end: str = "") -> pd.DataFrame:
    """
    FRED の時系列を取得し、index: date, column: 'value' の DataFrame を返す。
    """
    df = fred.get_series_df(name)[["date", "value"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.set_index("date")

    if end:
        df = df.loc[f"{start}":f"{end}"]
    else:
        df = df.loc[f"{start}":]
    return df


@st.cache_data
def load_fred_ohlcv(series_id: str, start: str = "2013-01-01", end: str = "") -> pd.DataFrame:
    """
    FRED の 'value' 時系列を擬似 OHLCV に変換。
    Open = High = Low = Close = value, Volume = 0 として返す。
    """
    df_val = get_fred_data(series_id, start=start, end=end)
    if df_val is None or df_val.empty or "value" not in df_val.columns:
        return pd.DataFrame()

    ohlcv = pd.DataFrame(index=df_val.index)
    for col in ["Open", "High", "Low", "Close"]:
        ohlcv[col] = df_val["value"]
    ohlcv["Volume"] = 0.0
    return ohlcv


# =====================================
#  Yahoo Finance / 株価系ヘルパー
# =====================================
def _flatten_stock_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance の戻り値（MultiIndex / 単一 Index 両方）を
    Date を index とする OHLCV DataFrame に整形。
    """
    if df_raw.empty:
        return df_raw

    df = df_raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # ("Close","AAPL") → "Close" だけに揃える
        df.columns = [col[0] for col in df.columns]

    df = df.reset_index().set_index("Date")
    df.index = pd.to_datetime(df.index)

    # 必要な列だけ残す
    cols_needed = ["Open", "High", "Low", "Close", "Volume"]
    cols_existing = [c for c in cols_needed if c in df.columns]
    return df[cols_existing]


@st.cache_data
def load_stock_df(ticker_symbol: str) -> pd.DataFrame:
    """
    yfinance でティッカーの株価データ（最大期間）を取得して整形。
    """
    df_raw = yf.download(tickers=ticker_symbol, period="max")
    return _flatten_stock_df(df_raw)


@st.cache_data
def load_stock_fundamentals(ticker_symbol: str) -> dict:
    """
    PER と EPS を yfinance.Ticker.info から一度に取得。
    """
    ticker = yf.Ticker(ticker_symbol)
    info = ticker.info

    return {
        "pe_trailing": info.get("trailingPE"),
        "pe_forward": info.get("forwardPE"),
        "eps_trailing": info.get("trailingEps"),
        "eps_forward": info.get("forwardEps"),
    }


@st.cache_data
def load_ratio_df_eem_efa() -> pd.DataFrame:
    """
    EEM/EFA の比率から疑似 OHLCV を作成。
    OHLC はそれぞれ EEM の OHLC / EFA の OHLC とする。
    Volume は 0。
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
            return pd.DataFrame()

    ratio_df["Volume"] = 0.0
    return ratio_df


# =====================================
#  共通ユーティリティ
# =====================================
def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    OHLCV を任意の頻度にまとめる。
    rule : '1H', '1D', '1W', '1M' など pandas resample ルール
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

    # OHLC のどれかがあれば OK として、全 NaN の期間を落とす
    price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df_resampled.columns]
    if price_cols:
        df_resampled = df_resampled.dropna(subset=price_cols)
    return df_resampled


def compute_ma_for_interval(df_resampled: pd.DataFrame) -> pd.DataFrame:
    """
    リサンプリング済 DataFrame に対し、Close の
    10/50/100/200 本分移動平均を計算。
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
    y軸レンジを「min → 10%, max → 80%」に配置するように計算。
    """
    if pd.isna(vmin) or pd.isna(vmax):
        return None

    span = vmax - vmin
    if span <= 0:
        if vmin == 0:
            return [0, 1]
        return [vmin * 0.8, vmax * 1.1]

    axis_span = span / 0.8
    axis_min = vmin - 0.1 * axis_span  # = vmin - span/8
    axis_max = axis_min + axis_span
    return [axis_min, axis_max]


def base_range_days(choice: str, full_start: pd.Timestamp, full_end: pd.Timestamp) -> float:
    """
    x_range_choice を日数に換算。
    """
    if choice == "3m":
        return 30.0 * 3.0
    if choice == "6m":
        return 30.0 * 6.0
    if choice == "1y":
        return 365.0
    if choice == "3y":
        return 365.0 * 3.0
    if choice == "5y":
        return 365.0 * 5.0
    if choice == "7y":
        return 365.0 * 7.0
    if choice == "10y":
        return 365.0 * 10.0
    if choice == "max":
        return max(1.0, (full_end - full_start).days)
    return 365.0  # デフォルト 1年


def get_index_description(symbol: str) -> str:
    """インデックス用説明テキストを取得。"""
    return INDEX_DESCRIPTION_MAP.get(symbol, "")


# =====================================
#  グラフ生成
# =====================================
def build_figure(
    df_input: pd.DataFrame,
    label: str,
    interval: str,
    base_range_choice: str,
    per: float | None = None,
    eps: float | None = None,
    show_volume: bool = True,
    show_per_in_title: bool = True,
    index_desc: str = "",
) -> go.Figure | None:
    """
    株価・インデックス共通の Plotly Figure を生成。
    """
    # 選択された interval でリサンプリング
    rule = FREQ_MAP[interval]
    df_freq = resample_ohlcv(df_input, rule)
    if df_freq.empty:
        # メインスレッド側でチェックしている前提だが、安全のため
        return None

    full_start = df_freq.index.min()
    full_end = df_freq.index.max()

    # 1d のときのベース期間（日数）
    base_days = base_range_days(base_range_choice, full_start, full_end)

    # interval に応じてスケーリング
    factor = INTERVAL_TO_DAYS.get(interval, 1.0) / INTERVAL_TO_DAYS["1d"]
    span_days = base_days * factor

    # 表示開始日
    start_candidate = full_end - pd.Timedelta(days=span_days)
    start = max(full_start, start_candidate)

    # 表示範囲 DataFrame
    df_view = df_freq.loc[(df_freq.index >= start) & (df_freq.index <= full_end)]
    if df_view.empty:
        df_view = df_freq.copy()
        start = df_view.index.min()

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

    # MA 計算（全期間）
    ma_for_plot = compute_ma_for_interval(df_freq)

    # 上昇/下落フラグ（全期間）
    up_all = df_plot["Close"] >= df_plot["Open"]
    volume_colors_all = ["green" if is_up else "red" for is_up in up_all]

    # MA 凡例ラベル
    unit = MA_UNIT_LABEL.get(interval, "")
    label_10 = f"MA 10{unit}" if unit else "MA 10"
    label_50 = f"MA 50{unit}" if unit else "MA 50"
    label_100 = f"MA 100{unit}" if unit else "MA 100"
    label_200 = f"MA 200{unit}" if unit else "MA 200"

    # ---------------------------
    # サブプロット構成
    # ---------------------------
    if show_volume:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.0,
            row_heights=[0.8, 0.2],
        )
        row_price, row_volume = 1, 2
    else:
        fig = make_subplots(
            rows=1,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.0,
            row_heights=[1.0],
        )
        row_price, row_volume = 1, None

    # ---------------------------
    # 価格（ローソク足 or 折れ線）
    # ---------------------------
    if show_volume:
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
        # 株価用：10/50/100 本
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
        # インデックス用：200本 MA のみ
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

    # ---------------------------
    # タイトル文字列
    # ---------------------------
    if show_per_in_title:
        # 株価用
        if per is not None and eps is not None:
            title_text = f"{label} PER:{per:.1f} EPS:{eps:.1f}"
        elif per is not None:
            title_text = f"{label} PER:{per:.1f}"
        else:
            title_text = f"{label} PER:NA"
    else:
        # インデックス用
        title_text = f"{label}: {index_desc}" if index_desc else label

    fig.add_annotation(
        x=0.5,
        y=1.0,
        xref="paper",
        yref="paper",
        text=title_text,
        showarrow=False,
        font=dict(size=16),
    )

    # レイアウト共通設定
    base_layout = dict(
        dragmode="pan",
        showlegend=True,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=0, b=0),
    )
    if show_volume:
        base_layout["height"] = 300
    else:
        base_layout["height"] = 200
    fig.update_layout(**base_layout)

    # x軸レンジ
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

    # y軸レンジ
    if price_range is not None:
        fig.update_yaxes(range=price_range, row=row_price, col=1)
    if show_volume and vol_range is not None and row_volume is not None:
        fig.update_yaxes(range=vol_range, row=row_volume, col=1)

    return fig


# =====================================
#  YTD 棒グラフ（並列ダウンロード）
# =====================================
def _ytd_worker(ticker: str):
    """
    YTD 計算用ワーカー（並列実行される）。
    """
    try:
        data = yf.download(ticker, start=YTD_START)
        if len(data) < 2:
            return None
        open_price = data.iloc[0]["Open"]
        current_price = data.iloc[-1]["Close"]
        ytd_val = float((current_price / open_price - 1) * 100)
        return {
            "Ticker": ticker,
            "YTD": ytd_val,
            "Country": COUNTRY_MAP_YTD[ticker],
        }
    except Exception:
        return None


# =====================================
#  YTD 棒グラフ
# =====================================
def build_ytd_figure() -> go.Figure | None:
    """
    YTD パフォーマンス棒グラフを作成。
    """
    results = []

    for ticker in YTD_TICKERS:
        data = yf.download(ticker, start=YTD_START)

        if len(data) < 2:
            st.warning(f"{ticker} のYTD計算に必要なデータが不足しています")
            continue

        open_price = data.iloc[0]["Open"]
        current_price = data.iloc[-1]["Close"]
        ytd_val = float((current_price / open_price - 1) * 100)

        results.append(
            {
                "Ticker": ticker,
                "YTD": ytd_val,
                "Country": COUNTRY_MAP_YTD[ticker],
            }
        )

    if not results:
        return None

    df_ytd = pd.DataFrame(results).sort_values("YTD", ascending=False)

    # x軸ラベル：Ticker と国名略称の2段表示
    x_labels = [f"{t}<br>{c}" for t, c in zip(df_ytd["Ticker"], df_ytd["Country"])]

    # 色付け（SPYだけ赤）
    colors = ["red" if t == "SPY" else "#1f567d" for t in df_ytd["Ticker"]]

    fig = go.Figure(
        data=[
            go.Bar(
                x=x_labels,
                y=df_ytd["YTD"],
                marker_color=colors,
                text=[f"{y:.2f}%" for y in df_ytd["YTD"]],
                textposition="outside",
            )
        ]
    )

    fig.update_layout(
        title="Year-to-Date Performance (YTD)",
        yaxis_title="YTD (%)",
        template="plotly_white",
        xaxis_tickfont=dict(size=12),
        margin=dict(l=40, r=40, t=80, b=50),
        height=400,
    )
    return fig


# =====================================
#  YTD 線グラフ
# =====================================
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

            data = yf.download(ticker, start=year_start, end=year_end, progress=False)

            # データ不足の場合はスキップ
            if len(data) < 2:
                print(f"{ticker} の {year} 年のデータが不足しています")
                continue

            open_price = data.iloc[0]["Open"]     # 年初の営業日の始値
            close_price = data.iloc[-1]["Close"]  # 年末の営業日の終値
            annual_return = float((close_price / open_price - 1) * 100)

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
        data = yf.download(ticker, start=ytd_start, end=ytd_end, progress=False)

        if len(data) < 2:
            print(f"{ticker} の {current_year} 年(YTD)のデータが不足しています")
            continue

        open_price = data.iloc[0]["Open"]     # 今年最初の営業日の始値
        close_price = data.iloc[-1]["Close"]  # 現在時点での直近営業日の終値
        ytd_return = float((close_price / open_price - 1) * 100)

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
        title=f"Annual Returns by Country ({start_year}–{current_year}) "
            f"(Last year: full-year, {current_year}: YTD)",
        xaxis_title="Year",
        yaxis_title="Return (%)",
        template="plotly_white",
        hovermode="x unified",
        legend_title="Country",
        margin=dict(l=40, r=40, t=80, b=40),
    )

    return fig


# =====================================
#  並列処理用ヘルパー（Stocks / Indexes）
# =====================================
def _load_stock_and_fundamentals(ticker: str):
    """
    Stocks 用: 株価データ + ファンダメンタルをまとめて取得（並列実行される）。
    """
    df = load_stock_df(ticker)
    fundamentals = load_stock_fundamentals(ticker)
    return ticker, df, fundamentals


def _load_index_ohlcv(idx_ticker: str):
    """
    Index (Yahoo Finance) 用: インデックスの OHLCV を取得（並列実行される）。
    """
    if idx_ticker == "EEM/EFA":
        df_idx = load_ratio_df_eem_efa()
    else:
        df_idx = load_stock_df(idx_ticker)
    return idx_ticker, df_idx


# =====================================
#  メイン（Streamlit UI）
# =====================================
def main():
    st.set_page_config(
        page_title="StockDash",
        layout="wide",
    )

    # ------------------------
    # サイドバー：Stock Controls
    # ------------------------
    st.sidebar.header("Stock Controls")

    n_charts = st.sidebar.number_input(
        "Number of stock tickers (n)",
        min_value=1,
        value=len(DEFAULT_STOCK_TICKERS),
        step=1,
    )

    interval = st.sidebar.radio(
        "Candle interval (box period)",
        options=list(FREQ_MAP.keys()),
        index=1,  # デフォルト: "1d"
    )

    x_range_choice = st.sidebar.radio(
        "X-axis window (for 1d)",
        options=X_RANGE_OPTIONS,
        index=2,  # デフォルト: "1y"
        horizontal=True,
    )

    # 株 Ticker 入力
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
    # サイドバー：Index Controls
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

    # ========================
    # 上部：Stocks （並列データ取得）
    # ========================
    st.markdown("## Stocks")

    # --- 並列で株価＆ファンダメンタル取得 ---
    stock_results: list[tuple[str, pd.DataFrame, dict]] = []
    max_workers = min(8, len(tickers))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(_load_stock_and_fundamentals, ticker): ticker
            for ticker in tickers
        }
        for future in as_completed(future_to_ticker):
            t = future_to_ticker[future]
            try:
                ticker, stock_df, fundamentals = future.result()
            except Exception as e:
                st.error(f"{t}: データ取得中にエラーが発生しました: {e}")
                continue
            stock_results.append((ticker, stock_df, fundamentals))

    # 元の tickers の順番にソートして表示順を維持
    stock_results.sort(key=lambda x: tickers.index(x[0]))

    n_cols = 4
    chart_idx = 0
    cols = None

    for ticker, stock_df, fundamentals in stock_results:
        if stock_df.empty:
            st.error(f"{ticker}: データが取得できませんでした。Ticker シンボルを確認してください。")
            continue

        per = fundamentals.get("pe_trailing")
        eps = fundamentals.get("eps_trailing")

        fig = build_figure(
            df_input=stock_df,
            label=ticker,
            interval=interval,
            base_range_choice=x_range_choice,
            per=per,
            eps=eps,
            show_volume=True,
            show_per_in_title=True,
            index_desc="",
        )
        if fig is None:
            continue

        if chart_idx % n_cols == 0:
            cols = st.columns(n_cols)
        col = cols[chart_idx % n_cols]
        with col:
            st.plotly_chart(fig, use_container_width=True)
        chart_idx += 1

    # ========================
    # 下部：Indexes（YF 部分は並列データ取得）
    # ========================
    st.markdown("## Indexes")

    index_tickers = DEFAULT_INDEX_TICKERS_YF + index_extra_tickers

    # --- 並列で YF インデックス取得 ---
    idx_results: list[tuple[str, pd.DataFrame]] = []
    if index_tickers:
        max_workers_idx = min(8, len(index_tickers))
        with ThreadPoolExecutor(max_workers=max_workers_idx) as executor:
            future_to_idx = {
                executor.submit(_load_index_ohlcv, idx_ticker): idx_ticker
                for idx_ticker in index_tickers
            }
            for future in as_completed(future_to_idx):
                idx_ticker = future_to_idx[future]
                try:
                    ticker_name, df_idx = future.result()
                except Exception as e:
                    st.error(f"{idx_ticker}: インデックスデータ取得中にエラーが発生しました: {e}")
                    continue
                idx_results.append((ticker_name, df_idx))

        # 表示順維持
        idx_results.sort(key=lambda x: index_tickers.index(x[0]))

    n_cols_idx = n_cols
    idx_chart = 0
    cols_idx = None

    # --- yfinance インデックス（並列で取得済み）---
    for idx_ticker, df_idx in idx_results:
        if df_idx.empty:
            st.error(f"{idx_ticker}: インデックスデータが取得できませんでした。Ticker シンボルを確認してください。")
            continue

        idx_desc = get_index_description(idx_ticker)

        fig_idx = build_figure(
            df_input=df_idx,
            label=idx_ticker,
            interval=interval,
            base_range_choice=x_range_choice,
            per=None,
            eps=None,
            show_volume=False,
            show_per_in_title=False,
            index_desc=idx_desc,
        )
        if fig_idx is None:
            continue

        if idx_chart % n_cols_idx == 0:
            cols_idx = st.columns(n_cols_idx)
        col_idx = cols_idx[idx_chart % n_cols_idx]
        with col_idx:
            st.plotly_chart(fig_idx, use_container_width=True)
        idx_chart += 1

    # --- FRED 系列インデックス（こちらは従来通り逐次でも、数が少ないので OK） ---
    for fred_series in DEFAULT_INDEX_FRED:
        df_fred = load_fred_ohlcv(fred_series)
        if df_fred.empty:
            st.warning(f"{fred_series}: FRED データが取得できませんでした。FRED シリーズIDや API キーを確認してください。")
            continue

        fred_desc = get_index_description(fred_series)

        fig_fred = build_figure(
            df_input=df_fred,
            label=fred_series,
            interval=interval,
            base_range_choice=x_range_choice,
            per=None,
            eps=None,
            show_volume=False,
            show_per_in_title=False,
            index_desc=fred_desc,
        )
        if fig_fred is None:
            continue

        if idx_chart % n_cols_idx == 0:
            cols_idx = st.columns(n_cols_idx)
        col_idx = cols_idx[idx_chart % n_cols_idx]
        with col_idx:
            st.plotly_chart(fig_fred, use_container_width=True)
        idx_chart += 1

    # ========================
    # その他：YTD バー（並列ダウンロード）
    # ========================
    st.markdown("## その他")

    fig_ytd = build_ytd_figure()
    fig_ytd_year = build_ytd_year_figure()
    if fig_ytd is not None:
        col_left, col_right = st.columns([1, 2])
        with col_left:
            st.plotly_chart(fig_ytd, use_container_width=True)
        with col_right:
            st.plotly_chart(fig_ytd_year, use_container_width=True)
    else:
        st.info("その他のYTDパフォーマンスを計算できるデータがありませんでした。")


if __name__ == "__main__":
    main()
