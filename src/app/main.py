"""Streamlit UI for stock expected-value screening tool."""

import hmac
import os
import sys
from pathlib import Path

# Add project root to sys.path for Streamlit Cloud compatibility
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import logging
from datetime import date, datetime, timedelta, timezone

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.app.pipeline_trigger import (
    fetch_workflow_runs,
    format_run_status,
    trigger_workflow,
)
from src.backtest.engine import load_backtest_results
from src.data.config import load_config
from src.data.ticker_master import load_ticker_master
from src.features.engineer import load_features
from src.ingest.financing_events import load_events
from src.overlay.integration import current_regime
from src.screening.screener import screen_candidates_v2

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


def _with_names(df: pd.DataFrame, ticker_master: pd.DataFrame | None) -> pd.DataFrame:
    """Join company names (company_name col, '---' when unknown)."""
    out = df.copy()
    if ticker_master is not None and not ticker_master.empty:
        out = out.merge(ticker_master[["ticker", "company_name"]], on="ticker", how="left")
        out["company_name"] = out["company_name"].fillna("---")
    else:
        out["company_name"] = "---"
    return out


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600)
def _load_all_data():
    """Load cached features, run layer-A-demoted screening with overlays.

    docs/07-10: the per-ticker EV ranking is invalid, so candidates come from
    ``screen_candidates_v2`` (pattern watchlist + avoidance overlay).  The old
    backtest cache is loaded only for the legacy detail/portfolio tabs and may
    be absent.
    """
    try:
        from src.data.release_store import ensure_data_available
        ensure_data_available()
    except Exception:
        logger.warning("release data fetch skipped", exc_info=True)
    features = load_features()
    if features is None:
        return None, None, None
    events = load_events()
    candidates = screen_candidates_v2(features, events)
    backtest = load_backtest_results()  # optional (legacy tabs)
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
    backtest: pd.DataFrame | None,
) -> None:
    """Render the main tab: reference watchlist + avoidance red flags (docs/07-10)."""
    st.info(
        "**このツールは「当てる」ものではなく「負けにくくする」もの**です。再検証(docs/07-09)で"
        "テクニカル期待値(旧+0.35%基準)は無効と判明したため、下表は**買い推奨ではなく観察用の"
        "ウォッチリスト**です。価値は ①毒性ファイナンスの**回避**(🚩) と ②地合いに応じた**リスク管理**(上部バナー) にあります。"
    )
    if candidates.empty:
        st.warning("本日、資金流入パターンに一致する銘柄はありません。")
        return

    screen_date = pd.Timestamp(candidates["date"].iloc[0])
    st.caption(f"スクリーニング日: {screen_date.strftime('%Y-%m-%d')}")

    flagged = candidates[candidates["red_flag"]] if "red_flag" in candidates else candidates.iloc[0:0]
    watch = candidates[~candidates["red_flag"]] if "red_flag" in candidates else candidates

    # --- avoidance red flags (layer B) ---
    if not flagged.empty:
        st.subheader("🚩 回避(毒性ファイナンス発表後・候補から除外)")
        st.caption(
            "行使価額修正条項付新株予約権/MSCB 発表から60日以内。検証08-09で中立後 +20日 -5%/+60日 -8%・"
            "45%が10%超下落。**買わない**。"
        )
        fdf = _with_names(flagged, ticker_master)
        ftab = pd.DataFrame({
            "銘柄コード": fdf["ticker"].values,
            "銘柄名": fdf["company_name"].values,
            "種別": fdf["flag_subtype"].values,
            "発表日": pd.to_datetime(fdf["flag_date"]).dt.strftime("%Y-%m-%d").values,
        })
        st.dataframe(ftab, use_container_width=True, hide_index=True)

    # --- reference watchlist (layer A demoted) ---
    st.subheader("資金流入パターン一致(参考ウォッチリスト)")
    st.caption("出来高急増+20日高値+陽線+ATR。出来高倍率順。※小型では「ブレイク後は追わない」警告として読む(検証09)。")
    wdf = _with_names(watch, ticker_master)
    wtab = pd.DataFrame({
        "銘柄コード": wdf["ticker"].values,
        "銘柄名": wdf["company_name"].values,
        "出来高倍率": wdf["turnover_ratio_5d"].round(2).values,
        "ATR比率": (wdf["atr14_ratio"] * 100).round(1).astype(str) + "%",
        "直近3日": (wdf["recent_3day_return"] * 100).round(1).astype(str) + "%",
    })
    st.dataframe(wtab, use_container_width=True, hide_index=True)

    # Ticker selection for legacy detail tab
    options = [f"{r['ticker']} - {n}" for r, n in
               zip(wdf.to_dict("records"), wdf["company_name"])]
    if options:
        selected = st.selectbox("詳細(レガシー)を表示する銘柄:", options)
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
    if not cand_row.empty and "recommended_hold_days" in cand_row.columns:
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
                entry_price = st.number_input("エントリー価格", min_value=0.0, value=0.0, step=1.0)

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
        bt_1d = backtest[(backtest["ticker"] == holding["ticker"]) & (backtest["hold_days"] == 1)]
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
# Tab 4: Screening analysis log
# ---------------------------------------------------------------------------


def _build_screening_log(
    features: pd.DataFrame,
    backtest: pd.DataFrame,
    config: dict,
    ticker_master: pd.DataFrame | None,
) -> dict:
    """Re-run screening steps and collect per-step statistics."""
    cfg = config
    ci = cfg["capital_inflow"]
    entry = cfg["entry"]
    exclusion = cfg["exclusion"]

    target_date = features["date"].max()
    df = features[features["date"] == target_date].copy()

    log: dict = {
        "target_date": target_date,
        "steps": [],
        "inflow_breakdown": [],
        "near_miss": pd.DataFrame(),
    }

    total = len(df)
    log["steps"].append(("全銘柄（対象日付）", total))

    if df.empty:
        return log

    # Liquidity filter
    df = df[df["liquidity_flag"]].copy()
    log["steps"].append(("流動性フィルター通過", len(df)))
    after_liquidity = len(df)

    if df.empty:
        return log

    # Capital inflow breakdown (individual condition counts)
    log["inflow_breakdown"] = [
        (
            "出来高倍率 >= {:.1f}".format(ci["turnover_ratio_5d_min"]),
            int((df["turnover_ratio_5d"] >= ci["turnover_ratio_5d_min"]).sum()),
            after_liquidity,
        ),
        ("20日高値ブレイク", int(df["high_20_break_flag"].sum()), after_liquidity),
        ("陽線 (close > open)", int((df["close"] > df["open"]).sum()), after_liquidity),
        (
            "ATR比率 >= {:.0%}".format(ci["atr14_ratio_min"]),
            int((df["atr14_ratio"] >= ci["atr14_ratio_min"]).sum()),
            after_liquidity,
        ),
        (
            "直近3日リターン <= {:.0%}".format(ci["recent_3day_return_max"]),
            int((df["recent_3day_return"] <= ci["recent_3day_return_max"]).sum()),
            after_liquidity,
        ),
    ]

    df = df[
        (df["turnover_ratio_5d"] >= ci["turnover_ratio_5d_min"])
        & (df["high_20_break_flag"])
        & (df["close"] > df["open"])
        & (df["atr14_ratio"] >= ci["atr14_ratio_min"])
        & (df["recent_3day_return"] <= ci["recent_3day_return_max"])
    ].copy()
    log["steps"].append(("資金流入フィルター通過（5条件AND）", len(df)))

    if df.empty:
        return log

    # Limit-up exclusion
    if exclusion["limit_up_next_day"]:
        df = df[((df["close"] - df["open"]) / df["open"]) < 0.15].copy()
        log["steps"].append(("ストップ高近似除外後", len(df)))

    if df.empty:
        return log

    # Backtest join
    bt_1d = backtest[backtest["hold_days"] == 1][
        ["ticker", "weighted_median_return", "weighted_win_rate", "weighted_dd_median"]
    ].copy()
    df = df.merge(bt_1d, on="ticker", how="inner")
    log["steps"].append(("バックテスト結合後", len(df)))

    if df.empty:
        return log

    # Build near-miss info before final filter
    def _ticker_name(t: str) -> str:
        if ticker_master is not None and not ticker_master.empty:
            m = ticker_master[ticker_master["ticker"] == t]
            if not m.empty:
                return m["company_name"].iloc[0]
        return "---"

    cols = ["ticker", "weighted_median_return", "weighted_win_rate", "weighted_dd_median"]
    near = df[cols].copy()
    near["銘柄名"] = near["ticker"].apply(_ticker_name)
    near["期待リターン判定"] = near["weighted_median_return"].apply(
        lambda x: "OK" if x >= entry["expected_return_1d_min"] else "NG"
    )
    near["勝率判定"] = near["weighted_win_rate"].apply(
        lambda x: "OK" if x >= entry["win_rate_min"] else "NG"
    )
    near["DD判定"] = near["weighted_dd_median"].apply(
        lambda x: "OK" if x >= entry["dd_median_max"] else "NG"
    )
    log["near_miss"] = near

    # Entry criteria
    df = df[
        (df["weighted_median_return"] >= entry["expected_return_1d_min"])
        & (df["weighted_win_rate"] >= entry["win_rate_min"])
        & (df["weighted_dd_median"] >= entry["dd_median_max"])
    ].copy()
    log["steps"].append(("エントリー基準通過", len(df)))

    log["entry_thresholds"] = entry
    return log


def render_log_tab(
    features: pd.DataFrame,
    backtest: pd.DataFrame,
    config: dict,
    ticker_master: pd.DataFrame | None,
) -> None:
    """Render the screening analysis log tab."""
    log = _build_screening_log(features, backtest, config, ticker_master)

    target_date = log["target_date"]
    st.caption(f"解析対象日: {pd.Timestamp(target_date).strftime('%Y-%m-%d')}")

    # Funnel table
    st.subheader("スクリーニング フィルター推移")
    funnel_data = []
    for i, (step_name, count) in enumerate(log["steps"]):
        prev = log["steps"][i - 1][1] if i > 0 else count
        drop = prev - count if i > 0 else 0
        funnel_data.append(
            {
                "ステップ": step_name,
                "残存銘柄数": count,
                "除外数": f"-{drop}" if drop > 0 else "—",
            }
        )
    st.table(pd.DataFrame(funnel_data))

    # Capital inflow breakdown
    if log["inflow_breakdown"]:
        st.subheader("資金流入条件（個別通過数）")
        breakdown_data = []
        for cond_name, passed, total in log["inflow_breakdown"]:
            breakdown_data.append(
                {
                    "条件": cond_name,
                    "通過数": f"{passed} / {total}",
                    "通過率": f"{passed / total * 100:.1f}%" if total > 0 else "—",
                }
            )
        st.table(pd.DataFrame(breakdown_data))

    # Near-miss details
    near = log.get("near_miss", pd.DataFrame())
    if not near.empty:
        st.subheader("エントリー基準の判定詳細")
        entry_th = log.get("entry_thresholds", {})
        if entry_th:
            st.caption(
                f"閾値: 期待リターン >= {entry_th.get('expected_return_1d_min', 0):+.2%}　"
                f"勝率 >= {entry_th.get('win_rate_min', 0):.0%}　"
                f"DD中央値 >= {entry_th.get('dd_median_max', 0):+.2%}"
            )
        display_near = pd.DataFrame()
        display_near["銘柄コード"] = near["ticker"].values
        display_near["銘柄名"] = near["銘柄名"].values
        display_near["期待リターン"] = near["weighted_median_return"].apply(lambda x: f"{x:+.3%}")
        display_near["期待リターン判定"] = near["期待リターン判定"].values
        display_near["勝率"] = near["weighted_win_rate"].apply(lambda x: f"{x:.1%}")
        display_near["勝率判定"] = near["勝率判定"].values
        display_near["DD中央値"] = near["weighted_dd_median"].apply(lambda x: f"{x:+.3%}")
        display_near["DD判定"] = near["DD判定"].values
        st.dataframe(display_near, use_container_width=True, hide_index=True)
    elif len(log["steps"]) > 0 and log["steps"][-1][1] == 0:
        # Ended before reaching entry criteria
        last_step = log["steps"][-1][0]
        st.info(f"「{last_step}」の時点で候補が0件のため、エントリー基準の判定は行われていません。")


# ---------------------------------------------------------------------------
# Tab 5: Pipeline trigger
# ---------------------------------------------------------------------------


def _get_secret(section: str, key: str) -> str:
    """Get a secret from Streamlit secrets or environment variable."""
    try:
        return st.secrets[section][key]
    except (KeyError, FileNotFoundError, AttributeError):
        env_key = f"{section.upper()}_{key.upper()}"
        return os.environ.get(env_key, "")


def render_pipeline_tab() -> None:
    """Render the pipeline trigger tab with password authentication."""
    admin_password = _get_secret("pipeline", "admin_password")
    github_pat = _get_secret("pipeline", "github_pat")

    if not admin_password or not github_pat:
        st.warning("パイプライン実行には Streamlit Secrets の設定が必要です。")
        st.code(
            "# .streamlit/secrets.toml\n"
            "[pipeline]\n"
            'admin_password = "your-password"\n'
            'github_pat = "ghp_xxxxx"  # actions:write 権限が必要',
            language="toml",
        )
        return

    # Password authentication
    password = st.text_input("管理パスワード", type="password", key="pipeline_password")
    if not password:
        st.info("パイプラインを実行するにはパスワードを入力してください。")
        return

    if not hmac.compare_digest(password, admin_password):
        st.error("パスワードが正しくありません。")
        return

    st.success("認証OK")

    # Workflow trigger buttons
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("夜間パイプライン")
        st.caption("データ取得 → 特徴量 → バックテスト → スクリーニング")
        if st.button("実行", key="trigger_nightly"):
            with st.spinner("トリガー中..."):
                result = trigger_workflow("nightly.yml", github_pat)
            if result["success"]:
                st.success(result["message"])
            else:
                st.error(result["message"])

    with col2:
        st.subheader("朝チェック")
        st.caption("ギャップチェック → 除外判定")
        if st.button("実行", key="trigger_morning"):
            with st.spinner("トリガー中..."):
                result = trigger_workflow("morning_check.yml", github_pat)
            if result["success"]:
                st.success(result["message"])
            else:
                st.error(result["message"])

    # Data refresh button
    st.divider()
    if st.button("データ再取得", key="refresh_data"):
        from src.data.release_store import ensure_data_available

        with st.spinner("GitHub Release からデータをダウンロード中..."):
            ensure_data_available(force_refresh=True)
        _load_all_data.clear()
        st.rerun()

    # Recent runs
    st.subheader("最近の実行履歴")
    hist_nightly, hist_morning = st.tabs(["夜間パイプライン", "朝チェック"])

    for hist_tab, wf_file in [(hist_nightly, "nightly.yml"), (hist_morning, "morning_check.yml")]:
        with hist_tab:
            runs = fetch_workflow_runs(wf_file, github_pat)
            if runs:
                for run in runs:
                    icon, label = format_run_status(run)
                    utc = datetime.fromisoformat(run["created_at"].replace("Z", "+00:00"))
                    jst = utc.astimezone(timezone(timedelta(hours=9)))
                    created = jst.strftime("%Y-%m-%d %H:%M")
                    st.write(f"{icon} {created} — {label}")
            else:
                st.info("実行履歴はありません。")


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
    st.title("小型株 回避・リスク管理スクリーナー")

    # Load data
    features, backtest, candidates = _load_all_data()

    # --- regime banner (layer C, docs/10) — always visible above tabs ---
    reg = current_regime()
    factor = f"推奨サイズ係数 ×{reg['size_factor']:.1f}"
    if reg["risk_off"]:
        st.error(f"🔴 地合い: {reg['label']}(GSPC<50日線・{reg['asof']})｜{factor}｜"
                 f"**新規スイングは抑制/縮小**(検証10: リスクオフ回避で最大DDを約1/3に圧縮)")
    else:
        st.success(f"🟢 地合い: {reg['label']}（{reg['asof']}）｜{factor}")

    # Five tabs — pipeline tab is always accessible
    tab_main, tab_detail, tab_portfolio, tab_log, tab_pipeline = st.tabs(
        ["📊 スクリーニング結果", "🔍 銘柄詳細", "💼 保有管理", "📋 解析ログ", "⚙️ パイプライン実行"]
    )

    _LEGACY = ("旧「期待値」ベースのタブです。再検証(docs/07-09)でテクニカル期待値は無効と判明したため"
               "無効化しています。価値は『スクリーニング結果』タブの回避(🚩)と上部の地合いバナーにあります。")

    if features is None:
        with tab_main:
            st.error("データキャッシュが見つかりません。以下を実行してください。")
            st.code("uv run python scripts/run_pipeline.py", language="bash")
    else:
        ticker_master = _load_ticker_master_safe()
        config = load_config()

        with tab_main:
            render_main_tab(candidates, ticker_master, backtest)

        with tab_detail:
            if backtest is not None:
                render_detail_tab(backtest, ticker_master, candidates)
            else:
                st.info(_LEGACY)

        with tab_portfolio:
            if backtest is not None:
                render_portfolio_tab(config, ticker_master, backtest, candidates)
            else:
                st.info(_LEGACY)

        with tab_log:
            if backtest is not None:
                render_log_tab(features, backtest, config, ticker_master)
            else:
                st.info(_LEGACY)

    with tab_pipeline:
        render_pipeline_tab()


if __name__ == "__main__":
    main()
