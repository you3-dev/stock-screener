# Streamlit 実装仕様書

作成日: 2026-02-21

## 1. ページ構成

### メイン画面

-   ランク
-   銘柄コード
-   1日期待値
-   勝率
-   推奨日数
-   DD

### 詳細ページ

-   日数別期待値グラフ
-   勝率グラフ
-   DD分布

### 保有管理

-   保有日数表示
-   強制決済カウント

## 2. 実行フロー

-   21:00 GitHub Actionsで更新
-   08:50 気配チェック処理

## 3. デプロイ

-   GitHub Repository
-   Streamlit Community Cloud

## 4. データ取得方式

Streamlit Community Cloud 上ではリポジトリに Parquet ファイルは含まれない（Git LFS 非対応）。

起動時に `src/data/release_store.py` の `ensure_data_available()` を呼び出し、GitHub Release（`data-latest` タグ）からデータをダウンロードして `data_cache/` に保存する。

-   GitHub Release API 経由で Parquet アセットを取得
-   公開リポジトリの場合はトークン不要
-   設定: `config/settings.yaml` の `release_store` セクション
