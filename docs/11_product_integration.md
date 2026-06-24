# 11 段D 製品統合 — 層B(回避)+層C(リスク管理)をアプリに実装

> 統合設計 docs/06 §8・段D。検証07-10で確定した「回避(層B)+リスク管理(層C)」を Streamlit 製品に載せ、
> 無効化された層A(テクニカル期待値)を降格。実施日: 2026-06-24。

## 実装内容

### 層C: 地合いバナー(全画面上部・常時)
- `src/overlay/integration.py::current_regime()` が最新レジームを返す(GSPC<50日線=リスクオフ、推奨サイズ係数)。
- アプリ上部に**🟢リスクオン / 🔴リスクオフ + 推奨サイズ係数(×1.0 / ×0.5)**を常時表示。リスクオフ時は
  「新規スイング抑制/縮小」を明示(検証10: リスクオフ回避で最大DDを約1/3)。**§1遵守(方向でなくリスクゲート)**。

### 層B: 回避オーバーレイ(メインタブ)
- `src/screening/screener.py::screen_candidates_v2()` が資金流入パターン一致銘柄を出し、
  `src/overlay/avoidance.py` で **Tier A(行使価額修正/MSCB)発表後60日以内を候補から除外+🚩表示**。
- メインタブに「🚩回避(除外済)」セクション(種別・発表日)と「参考ウォッチリスト」を分離表示。

### 層A: 降格
- per-ticker in-sample 期待値ランキング(+0.35%基準)・旧 `screen_candidates`・旧 backtest 依存を**メイン経路から除去**。
- メインタブ冒頭に明示:「**当てるのでなく負けにくくするツール。下表は買い推奨でなく観察用ウォッチリスト**」。
- 旧「期待値」ベースのタブ(銘柄詳細/解析ログ等)は**レガシー注記に置換**(再検証で無効化、docs/07-09参照)。

### データモデルの更新
- `src/data/release_store.py`: 同期対象から `backtest_results.parquet` を除外し、
  **`financing_events.parquet`(層B)+ `macro.parquet`(層C)を追加**。
- `scripts/run_pipeline.py`(層A降格版): prices→**features + 毒性ファイナンスevents(seed)+ macro/regime** を生成。
  旧 backtest は生成しない。

## スモークテスト(Streamlit AppTest・ヘッドレス)
```
uv run python scripts/run_pipeline.py            # features/events/macro 生成
uv run python -c "from streamlit.testing.v1 import AppTest; at=AppTest.from_file('src/app/main.py'); at.run(); assert not at.exception"
```
- **結果: 例外0で起動**。5タブ、地合いバナー(リスクオン=緑)、メインタブにウォッチリスト(本日2件・赤旗0=
  直近60日にTier A無し)、レガシータブは注記。現データ: 3,739銘柄・events 1,243(Tier A 567)・regime asof 2026-06-23 リスクオン。

## 残りの本番化タスク(follow-up)
1. **GitHub Actions**(このcloneには未配置): 夜間に `fetch_prices.py → run_pipeline.py → financing_events.ingest_forward`
   を実行し、生成キャッシュを Release(`data-latest`)へ上げる。`release_store` は新ファイル一覧に更新済み。
2. **層Bの前進取込**: `financing_events.ingest_forward(from, to)` を日次に(yanoshin、キー不要、3秒スリープ)。
   過去分は finance DB から seed 済み(2023-09〜2026-02)。
3. **保有管理タブの作り直し**: 旧EV依存を外し、地合いサイズ係数・保有日数/強制決済カウントダウンの
   リスク管理ツールとして再実装(層Cと連動)。今回はレガシー注記で一旦退避。
4. `use_container_width` の非推奨置換(`width='stretch'`)など軽微なUI追従。

## まとめ
**「ブレイク候補を当てて買う」スクリーナーから、「小型の地雷を回避(B)し地合いでリスクを落とす(C)」
負けにくさツールへ**、製品レベルで転換完了。検証07-11で設計(docs/06)を実データで裏付け、UIに実装した。
