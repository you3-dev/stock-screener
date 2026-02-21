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
