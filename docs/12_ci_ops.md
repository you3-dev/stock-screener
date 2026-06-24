# 12 CI / 運用(GitHub Actions)

> 設計 docs/06 §8 の本番化。夜間バッチと朝チェックを GitHub Actions 化し、生成データを
> GitHub Release(`data-latest`)へ配信、Streamlit Cloud が起動時にDLする。実装: 2026-06-24。

## データのライフサイクル

```
GitHub Actions(夜間/朝)
  → data_cache/*.parquet を生成
  → GitHub Release "data-latest" に --clobber でアップロード
Streamlit Cloud(起動時)
  → src/data/release_store.ensure_data_available() が Release からDL
```
**リポジトリにはデータを置かない**(`.gitignore` で `data_cache/`。docs/11)。Release が唯一の配信元。

## ワークフロー

### `.github/workflows/nightly.yml`(21:00 JST = 12:00 UTC 平日)
1. 既存 `financing_events.parquet` を Release から取得(履歴の継ぎ足し用・初回は無し)
2. `scripts/fetch_prices.py` — 全3市場(Prime/Standard/Growth)・adj_close保持・再開可能
3. `scripts/run_pipeline.py` — 特徴量 + マクロ/レジーム(旧backtestは生成しない=層A降格)
4. `scripts/ingest_events.py --days-back N` — yanoshin 前進取込(層B、キー不要、3秒スリープ)
5. `data-latest` へ5ファイル(prices/features/ticker_master/financing_events/macro)を upload

### `.github/workflows/morning_check.yml`(08:50 JST = 23:50 UTC 前日 平日)
- `scripts/morning_check.py` — レジーム更新(地合いバナー)+ 当日ウォッチリストの軽量ギャップチェック
- `macro.parquet`(地合い)と `market_check.txt` を Release に反映

## 権限・シークレット
- **スケジュール実行**: 組み込みの `GITHUB_TOKEN` + `permissions: contents: write` のみ(追加設定不要)。
- **アプリからの手動トリガー**(パイプラインタブ): Streamlit Secrets に PAT が必要(既存)。
  ```toml
  [pipeline]
  admin_password = "..."
  github_pat = "ghp_..."   # actions:write
  ```
- **Streamlit Cloud のデータDL**: 公開リポなら不要。プライベート/レート対策で必要なら Secrets に
  `GITHUB_TOKEN`(repo read)を設定(`release_store._get_token` が参照)。

## 初回ブートストラップ
1. `nightly.yml` を **手動実行(workflow_dispatch)し、`events_days_back=90`** にする
   → 赤旗の60日窓を満たす毒性ファイナンス履歴が貯まる(以降は毎晩14日分を継ぎ足し=重複はdedup)。
2. 以降は夜間/朝が自動で Release を更新。Streamlit Cloud は再起動で最新を取得。

## 補足
- **過去2.5年の毒性ファイナンス履歴**(finance DB seed)はバックテスト用でローカルのみ。
  ライブの赤旗は**直近60日**だけ見るため、前進取込で十分(履歴のRelease投入は不要)。
- 夜間の価格は毎回フル取得(~3,739銘柄)。重い場合は将来 incremental 化を検討(現状は正確性優先)。
- `fetch_prices.py` は失敗チャンクを次回再試行(再開可能)。yfinance 一時障害でも翌晩回復。
