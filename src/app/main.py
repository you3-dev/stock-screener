"""Streamlit UI for stock expected-value screening tool."""

import logging
from datetime import date

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.backtest.engine import load_backtest_results
from src.data.config import load_config
from src.data.ticker_master import load_ticker_master
from src.features.engineer import load_features
from src.screening.screener import screen_candidates

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions (pure, testable)
# ---------------------------------------------------------------------------


def compute_total_score(df: pd.DataFrame) -> pd.Series:
    """Compute total score (0-100) from return, win_rate, dd_median.

    Score = 0.4 * norm(return) + 0.3 * norm(win_rate) + 0.3 * norm(dd_median)
    Each component is min-max normalized to 0-100.
    """

    def _minmax(s: pd.Series) -> pd.Series:
        rng = s.max() - s.min()
        if rng == 0:
            return pd.Series(50.0, index=s.index)
        return (s - s.min()) / rng * 100

    norm_ret = _minmax(df["expected_return_1d"])
    norm_wr = _minmax(df["win_rate"])
    norm_dd = _minmax(df["dd_median"])  # closer to 0 is better, so higher = better

    return (0.4 * norm_ret + 0.3 * norm_wr + 0.3 * norm_dd).round(1)


def compute_hold_days(entry_date: date, today: date | None = None) -> int:
    """Compute business days held since entry_date."""
    if today is None:
        today = date.today()
    return int(np.busday_count(entry_date, today))


def compute_remaining_days(hold_days: int, max_hold_days: int) -> int:
    """Compute remaining business days before forced exit."""
    return max_hold_days - hold_days


def format_display_table(
    candidates: pd.DataFrame, ticker_master: pd.DataFrame | None
) -> pd.DataFrame:
    """Build display table by joining company names and adding rank."""
    if candidates.empty:
        return pd.DataFrame(
            columns=[
                "ランク", "銘柄コード", "銘柄名", "1日期待値",
                "勝率", "推奨日数", "最大DD", "総合スコア",
            ]
        )

    df = candidates.copy()

    # Join company name
    if ticker_master is not None and not ticker_master.empty:
        df = df.merge(
            ticker_master[["ticker", "company_name"]], on="ticker", how="left"
        )
        df["company_name"] = df["company_name"].fillna("---")
    else:
        df["company_name"] = "---"

    # Compute total score
    df["total_score"] = compute_total_score(df)

    # Build display table
    display = pd.DataFrame()
    display["ランク"] = range(1, len(df) + 1)
    display["銘柄コード"] = df["ticker"].values
    display["銘柄名"] = df["company_name"].values
    display["1日期待値"] = df["expected_return_1d"].values
    display["勝率"] = df["win_rate"].values
    display["推奨日数"] = df["recommended_hold_days"].astype(int).values
    display["最大DD"] = df["dd_median"].values
    display["総合スコア"] = df["total_score"].values

    return display


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600)
def _load_all_data():
    """Load cached features and backtest results, then run screening."""
    features = load_features()
    backtest = load_backtest_results()
    if features is None or backtest is None:
        return None, None, None
    candidates = screen_candidates(features, backtest)
    return features, backtest, candidates


def _load_ticker_master_safe() -> pd.DataFrame | None:
    """Load ticker master with error handling."""
    try:
        return load_ticker_master()
    except Exception:
        logger.warning("Failed to load ticker master", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Tab 1: Main screening results
# ---------------------------------------------------------------------------


def render_main_tab(
    candidates: pd.DataFrame,
    ticker_master: pd.DataFrame | None,
    backtest: pd.DataFrame,
) -> None:
    """Render the main screening results tab."""
    if candidates.empty:
        st.info("本日のスクリーニング候補はありません。")
        return

    # Show screening date
    screen_date = candidates["date"].iloc[0]
    st.caption(f"スクリーニング日: {screen_date.strftime('%Y-%m-%d')}")

    # Build display table
    display = format_display_table(candidates, ticker_master)

    # DD warning
    dd_warnings = candidates[candidates["dd_median"] < -0.03]
    if not dd_warnings.empty:
        tickers = ", ".join(dd_warnings["ticker"].tolist())
        st.warning(f"DD が -3% を超える銘柄があります: {tickers}")

    # Format percentages for display
    styled = display.copy()
    styled["1日期待値"] = styled["1日期待値"].apply(lambda x: f"{x:+.2%}")
    styled["勝率"] = styled["勝率"].apply(lambda x: f"{x:.1%}")
    styled["最大DD"] = styled["最大DD"].apply(lambda x: f"{x:+.2%}")

    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Ticker selection for detail tab
    st.subheader("銘柄選択")
    options = []
    for _, row in display.iterrows():
        options.append(f"{row['銘柄コード']} - {row['銘柄名']}")

    if options:
        selected = st.selectbox("詳細を表示する銘柄を選択:", options)
        if selected:
            st.session_state["selected_ticker"] = selected.split(" - ")[0]


# ---------------------------------------------------------------------------
# Tab 2: Ticker detail
# ---------------------------------------------------------------------------


def render_detail_tab(
    backtest: pd.DataFrame,
    ticker_master: pd.DataFrame | None,
    candidates: pd.DataFrame,
) -> None:
    """Render the ticker detail tab with charts."""
    selected = st.session_state.get("selected_ticker")
    if not selected:
        st.info("メイン画面から銘柄を選択してください。")
        return

    # Get company name
    company_name = "---"
    if ticker_master is not None and not ticker_master.empty:
        match = ticker_master[ticker_master["ticker"] == selected]
        if not match.empty:
            company_name = match["company_name"].iloc[0]

    st.subheader(f"{selected} - {company_name}")

    # Get backtest data for this ticker
    bt = backtest[backtest["ticker"] == selected].sort_values("hold_days").copy()
    if bt.empty:
        st.warning("この銘柄のバックテスト結果がありません。")
        return

    # Get recommended hold days from candidates
    cand_row = candidates[candidates["ticker"] == selected]
    recommended_n = None
    if not cand_row.empty:
        recommended_n = int(cand_row["recommended_hold_days"].iloc[0])

    # Sample size metric
    sample = bt[bt["hold_days"] == 1]["sample_size"]
    if not sample.empty:
        st.metric("類似パターン件数", f"{sample.iloc[0]:,}")

    # Chart data
    bt["return_pct"] = bt["weighted_median_return"] * 100
    bt["winrate_pct"] = bt["weighted_win_rate"] * 100
    bt["dd_pct"] = bt["weighted_dd_median"] * 100

    col1, col2 = st.columns(2)

    with col1:
        # Expected return chart
        st.caption("日数別 期待リターン (%)")
        base = alt.Chart(bt).encode(
            x=alt.X("hold_days:O", title="保有日数"),
            y=alt.Y("return_pct:Q", title="期待リターン (%)"),
        )
        bars = base.mark_bar(color="#4C78A8")
        chart = bars
        if recommended_n is not None:
            rule_data = pd.DataFrame({"hold_days": [recommended_n]})
            rule = (
                alt.Chart(rule_data)
                .mark_rule(color="red", strokeWidth=2, strokeDash=[4, 4])
                .encode(x="hold_days:O")
            )
            chart = bars + rule
        st.altair_chart(chart, use_container_width=True)

    with col2:
        # Win rate chart
        st.caption("日数別 勝率 (%)")
        cfg = load_config()
        wr_threshold = cfg["entry"]["win_rate_min"] * 100
        base_wr = alt.Chart(bt).encode(
            x=alt.X("hold_days:O", title="保有日数"),
            y=alt.Y("winrate_pct:Q", title="勝率 (%)"),
        )
        bars_wr = base_wr.mark_bar(color="#72B7B2")
        rule_wr = (
            alt.Chart(pd.DataFrame({"y": [wr_threshold]}))
            .mark_rule(color="orange", strokeWidth=2, strokeDash=[4, 4])
            .encode(y="y:Q")
        )
        st.altair_chart(bars_wr + rule_wr, use_container_width=True)

    # DD chart (full width)
    st.caption("日数別 最大DD (%)")
    base_dd = alt.Chart(bt).encode(
        x=alt.X("hold_days:O", title="保有日数"),
        y=alt.Y("dd_pct:Q", title="最大DD (%)"),
    )
    bars_dd = base_dd.mark_bar(color="#E45756")
    rule_dd = (
        alt.Chart(pd.DataFrame({"y": [-3.0]}))
        .mark_rule(color="red", strokeWidth=2, strokeDash=[4, 4])
        .encode(y="y:Q")
    )
    st.altair_chart(bars_dd + rule_dd, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 3: Portfolio management
# ---------------------------------------------------------------------------


def render_portfolio_tab(
    config: dict,
    ticker_master: pd.DataFrame | None,
    backtest: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    """Render the portfolio management tab."""
    max_holdings = config["portfolio"]["max_holdings"]
    max_hold_days = config["portfolio"]["max_hold_days"]

    # Initialize portfolio in session state
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = []

    portfolio = st.session_state.portfolio

    # Add holding form
    st.subheader("銘柄追加")
    if len(portfolio) >= max_holdings:
        st.warning(f"最大保有数（{max_holdings}銘柄）に達しています。")
    else:
        with st.form("add_holding", clear_on_submit=True):
            # Build ticker options from candidates
            ticker_options = []
            if not candidates.empty:
                for t in candidates["ticker"].tolist():
                    name = "---"
                    if ticker_master is not None:
                        match = ticker_master[ticker_master["ticker"] == t]
                        if not match.empty:
                            name = match["company_name"].iloc[0]
                    ticker_options.append(f"{t} - {name}")

            col1, col2, col3 = st.columns(3)
            with col1:
                if ticker_options:
                    selected_str = st.selectbox("銘柄", ticker_options)
                else:
                    selected_str = st.text_input("銘柄コード", placeholder="7203.T")
            with col2:
                entry_date = st.date_input("エントリー日", value=date.today())
            with col3:
                entry_price = st.number_input(
                    "エントリー価格", min_value=0.0, value=0.0, step=1.0
                )

            submitted = st.form_submit_button("追加")
            if submitted and selected_str:
                ticker_code = selected_str.split(" - ")[0].strip()
                company = "---"
                if ticker_master is not None:
                    match = ticker_master[ticker_master["ticker"] == ticker_code]
                    if not match.empty:
                        company = match["company_name"].iloc[0]
                portfolio.append(
                    {
                        "ticker": ticker_code,
                        "company_name": company,
                        "entry_date": str(entry_date),
                        "entry_price": entry_price,
                    }
                )
                st.rerun()

    # Holdings list
    st.subheader("保有一覧")
    if not portfolio:
        st.info("保有銘柄はありません。")
        return

    for i, holding in enumerate(portfolio):
        entry_d = date.fromisoformat(holding["entry_date"])
        hold = compute_hold_days(entry_d)
        remaining = compute_remaining_days(hold, max_hold_days)

        # Status
        if remaining <= 0:
            status = "🔴 強制決済"
        elif remaining <= 3:
            status = "⚠️ 残りわずか"
        else:
            status = "✅ 保有中"

        # Next-day expected return from backtest
        bt_1d = backtest[
            (backtest["ticker"] == holding["ticker"]) & (backtest["hold_days"] == 1)
        ]
        next_day_ret = bt_1d["weighted_median_return"].iloc[0] if not bt_1d.empty else None

        col1, col2, col3, col4, col5, col6 = st.columns([2, 2, 1, 1, 1, 1])
        with col1:
            st.write(f"**{holding['ticker']}** - {holding['company_name']}")
        with col2:
            st.write(f"エントリー: {holding['entry_date']}")
        with col3:
            st.write(f"保有: {hold}日")
        with col4:
            st.write(f"残り: {remaining}日")
        with col5:
            if next_day_ret is not None:
                st.write(f"翌日期待値: {next_day_ret:+.2%}")
            else:
                st.write("翌日期待値: ---")
        with col6:
            st.write(status)

        if st.button("削除", key=f"del_{i}"):
            portfolio.pop(i)
            st.rerun()

        st.divider()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Main Streamlit application."""
    st.set_page_config(
        page_title="株価期待値スクリーニング",
        page_icon="📈",
        layout="wide",
    )
    st.title("株価期待値スクリーニングツール")

    # Load data
    features, backtest, candidates = _load_all_data()
    if features is None:
        st.error("データキャッシュが見つかりません。先にパイプラインを実行してください。")
        st.code("uv run python scripts/run_pipeline.py", language="bash")
        return

    ticker_master = _load_ticker_master_safe()
    config = load_config()

    # Three tabs
    tab_main, tab_detail, tab_portfolio = st.tabs(
        ["📊 スクリーニング結果", "🔍 銘柄詳細", "💼 保有管理"]
    )

    with tab_main:
        render_main_tab(candidates, ticker_master, backtest)

    with tab_detail:
        render_detail_tab(backtest, ticker_master, candidates)

    with tab_portfolio:
        render_portfolio_tab(config, ticker_master, backtest, candidates)


if __name__ == "__main__":
    main()
