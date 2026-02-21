# 詳細データ設計書

作成日: 2026-02-21

## 1. 銘柄マスタ

-   ticker
-   company_name
-   market (prime限定)

## 2. 日足価格データ

-   date
-   ticker
-   open
-   high
-   low
-   close
-   volume
-   turnover

## 3. 特徴量テーブル

-   date
-   ticker
-   turnover_ratio_5d
-   atr14_ratio
-   high_20_break_flag
-   recent_3day_return
-   liquidity_flag

## 4. バックテスト結果テーブル

-   ticker
-   hold_days
-   weighted_median_return
-   weighted_win_rate
-   weighted_dd_median
-   sample_size

## 5. エントリー候補テーブル（毎日更新）

-   date
-   ticker
-   expected_return_1d
-   win_rate
-   recommended_hold_days
-   dd_median

## 6. データ保存方式

### GitHub Release アセット

パイプライン（GitHub Actions 夜間バッチ）で生成した Parquet ファイルは、GitHub Release（`data-latest` タグ）のアセットとしてアップロードする。

-   保存先: `gh release upload data-latest <file> --clobber`
-   対象ファイル: prices.parquet, features.parquet, backtest_results.parquet, ticker_master.parquet
-   設定: `config/settings.yaml` の `release_store.repo` / `release_store.tag`

### Streamlit Cloud からの取得

Streamlit Community Cloud 起動時に `src/data/release_store.py` の `ensure_data_available()` を呼び出し、ローカルキャッシュ（`data_cache/`）にファイルがなければ GitHub Release API からダウンロードする。

-   Git LFS は Streamlit Cloud 非対応のため不使用
-   `data_cache/` は `.gitignore` に含めリポジトリにはコミットしない
