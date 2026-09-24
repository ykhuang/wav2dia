# Wav2Dia TUI 開發規格

**狀態：** Files / Setup 導覽、依賴與 Hugging Face 憑證檢查、真實 transcription → diarization 背景工作、取消與續跑狀態、Results 預覽均已實作。
**目標環境：** WSL / Ubuntu 終端機、Python 3.11 或 3.12。
**UI 框架：** Textual。

## 1. 目標與範圍

TUI 是 `wav2dia` CLI 的工作區前端：從 `files/` 掃描媒體、啟動本機語音辨識與 speaker diarization、管理可續跑狀態，以及預覽產物。

目前不包含音訊播放、逐字字幕編輯、TUI 內 `config.toml` 寫入、LLM backend 實際健康檢查，或右欄實質功能。

```text
workspace/
├── files/                         # 輸入媒體，遞迴掃描
│   └── meeting.mp3
├── output/
│   └── meeting/
│       ├── meeting.raw.srt         # 完成 ASR 後保留
│       ├── meeting.raw.md
│       ├── meeting.raw.segments.json
│       ├── meeting.srt             # speaker-aligned 最終字幕
│       ├── meeting.md
│       ├── meeting.segments.json
│       ├── meeting.speakers.txt
│       └── meeting.run-status.json # 本次／前次執行狀態
├── config.toml                     # 可選
├── .env                            # 可選，且必須不納入版本控制
├── .env.example
└── TUI_SPEC.md
```

支援 `.wav`、`.mp3`，以及 pipeline 已支援的 `.m4a`、`.flac`、`.mp4`、`.mkv`、`.webm`。

## 2. 版面與焦點

畫面採上三、下一：

```text
┌────────────────────┬──────────────────────────────────┬────────────────────┐
│ Files / Setup      │ Transcription results            │ Reserved           │
│                    │                                  │                    │
│ Files              │ ▸ meeting.wav                    │ (MVP 空白)         │
│   meeting.wav      │                                  │                    │
│ Setup              │                                  │                    │
│   Diarize min=3... │                                  │                    │
│   ▸ Dependency     │                                  │                    │
│   Hugging Face ... │                                  │                    │
└────────────────────┴──────────────────────────────────┴────────────────────┘
┌────────────────────────────────────────────────────────────────────────────┐
│ Status / preview                                                            │
│ SRT 或 Markdown 內容、設定摘要、工作階段、錯誤或動作提示                    │
└────────────────────────────────────────────────────────────────────────────┘
```

- 左、中、右欄寬度為 `37% / 37% / 26%`；左、中欄最小寬度分別為 24、36。
- 下欄最小高度為 10 列；可捲動。
- `Tab`／`Shift+Tab` 切換焦點；含焦點子元件的 pane 使用高對比黃色粗外框，焦點 Tree 的游標也提高對比。
- 右欄僅保留版面，不應成為 MVP 的工作流程依賴。

## 3. Files 與 Setup

### Files

- 遞迴列出 `files/` 下受支援媒體；資料夾可按 `Enter` 展開或收合。
- 游標停在媒體時，下欄顯示路徑、大小、時長與動作提示。
- `F5` 重新掃描 `files/` 與 `output/`，並重新執行快速啟動檢查。
- 媒體按 `Enter` 時先開啟含時長與有效 speaker 範圍的確認框，再依既有 output 狀態開始工作或開啟動作選擇框。

### Setup

Setup 預設展開，依序顯示：

1. **Diarize**：顯示本 TUI session 的 `min`、`max` speaker 範圍；按 `Enter` 開啟設定框。
2. **Dependency**：按 `Enter` 展開。
3. **Hugging Face access**：獨立於 Dependency 顯示。

#### Diarize

- 預設 `min_speakers=3`、`max_speakers=6`。
- 設定框只接受整數，且必須符合 `min_speakers >= 1`、`max_speakers >= min_speakers`。
- 選擇 **OK** 後立即套用至此 TUI session 後續啟動的工作；不改寫 `config.toml`。
- 開始工作前的確認框會再次顯示選取音檔時長與這兩個有效值。

#### Dependency

| 項目 | 啟動時檢查 | 失敗呈現 |
| --- | --- | --- |
| `faster-whisper>=1.1,<2` | 已安裝 distribution 與版本範圍 | 紅色 |
| `pyannote.audio>=4,<5` | 已安裝 distribution 與版本範圍 | 紅色 |
| `textual>=0.58` | 已安裝 distribution 與版本範圍 | 紅色 |
| `mutagen>=1.47` | 已安裝 distribution 與版本範圍 | 紅色 |
| `ffmpeg` | `ffmpeg -version` | 紅色 |

- 任一子項失敗，`Dependency` 父項也為紅色。
- 啟動檢查**不得**匯入 `faster_whisper`、`pyannote.audio` 或 `torch`；只讀取 package metadata，以避免 PyTorch 延遲 TUI 啟動。
- 真正工作開始後才載入模型；匯入、CUDA、decoder 或網路錯誤在工作失敗狀態顯示。

#### Hugging Face access

- 讀取順序：既有 shell 環境變數 → 工作區根目錄 `.env`。
- 預設變數名稱是 `HF_TOKEN`，實際名稱依 `config.toml` 的 `diarization.token_env`。
- 僅顯示 token 是否存在與來源（environment／`.env`），不得顯示 token 值。
- 缺少 token 時為紅色，並阻擋任何包含 diarization 的工作。
- token 存在不保證已接受 pyannote 模型條款；實際模型存取時的權限錯誤須顯示為工作失敗。

## 4. Transcription results 與內容預覽

中欄依 `files/` 來源的相對路徑分組。對群組按 `Enter` 展開或收合：

```text
▾ interviews/day1/meeting.wav
  • ✓ completed (diarized)
  • srt
  • md
  • txt (speaker turns)
```

- 游標移到 `.srt` 或 `.md` 時，下欄讀取完整 UTF-8 內容並預覽。
- `.txt` 目前可選取並顯示基本資訊；speaker turns 原文預覽不在目前範圍。
- raw 中介檔預設不在完整結果樹顯示；若取消／失敗後 ASR 已完成，顯示可用的 raw SRT／MD，及橘色的 `txt (speaker turns — incomplete)` 虛擬項目。
- `segments.json` 是 canonical 資料，保留於磁碟供續跑，不佔正常結果樹位置。

### 結果顏色

| 狀態 | 顏色 | 意義 |
| --- | --- | --- |
| `completed` 且來源相符 | 預設 | 完整可用結果 |
| `cancelled`、`abandoned`、`running` | 橘色 | 前次／目前工作未完整結束；可能可續跑 |
| `failed`、`invalid`、來源指紋不符 | 紅色 | 需要使用者處理；不可安全續跑 |

目前輸出路徑以主檔名建立。例如同一資料夾內 `sample.wav` 與 `sample.mp3` 都會解析到 `output/sample/`。狀態 JSON 的來源指紋會將另一個來源標為紅色 `needs attention`，避免錯用其結果；使用者應改名或移至不同子資料夾。

## 5. 真實 Pipeline、取消與續跑

### 預設工作

在媒體上按 `Enter`：

1. 顯示確認框：音檔名稱、時長、有效的 `min_speakers` 與 `max_speakers`。
2. 使用者選擇 **OK** 才快速檢查 Dependency 與 Hugging Face access；選擇 **Cancel** 不開始工作。
3. 判定是否存在可信的前次狀態。
4. 需要時讓使用者選擇動作。
5. 以 Textual background worker 執行 `Transcribing → Saving raw transcript → Diarizing → Saving speaker-aligned results`。
6. 完成、取消或失敗時寫入狀態、關閉進度框並重新整理 Results。

同時僅允許一個 pipeline 工作。

### 進度框

- 視窗寬度為終端機的 `60vw`，水平與垂直置中。
- 顯示目前階段、ASCII spinner、循環式不定進度條與累積 `Elapsed time`。
- ASR 與 diarization 無可靠百分比時，進度條每 10 秒前進一格，填滿後循環。

### 取消

- `Ctrl+C` 是「取消請求」，不是立即強殺 thread。
- 模型呼叫是阻塞式；畫面先顯示 `Cancelling…`，取消在目前模型階段返回後生效。
- 在 transcription 中取消：不寫完整 raw transcript，也不進入 diarization。
- 在 raw transcript 已完成後、diarization 中取消：保留 raw artifacts，不寫最終 speaker-aligned 結果。
- 使用 `q`／`Ctrl+Q` 離開而未完成的 `running` 紀錄，在下次選取該檔案時視為 `abandoned`。

### Run status sidecar

每次工作寫入 `<basename>.run-status.json`，至少包含：

```json
{
  "schema_version": 1,
  "status": "cancelled",
  "last_completed_stage": "transcribed",
  "mode": "full",
  "source": {
    "path": "/absolute/path/to/meeting.wav",
    "size": 123,
    "mtime_ns": 123
  },
  "message": "Cancelled during diarization.",
  "raw_artifacts": {"json": "...raw.segments.json"}
}
```

SRT、Markdown、speaker TXT 不插入 `cancelled by user` 文字，避免破壞格式。所有 artifact set 先寫入同資料夾的暫存檔，再以原子替換提交正式檔。

### 已存在輸出時的選項

| 前次狀態 | 選項 |
| --- | --- |
| 無輸出 | 直接完整執行 |
| `completed` 且來源指紋相符 | `Overwrite / Cancel` |
| `cancelled`／`failed`，且有相符來源與完整 `*.raw.segments.json` | `Overwrite / Skip to diarize / Cancel` |
| transcription 未完成 | `Overwrite / Cancel` |
| 來源指紋不同、sidecar 無效、或舊輸出沒有 sidecar | `Overwrite / Cancel` |

`Skip to diarize` 只讀取 `*.raw.segments.json`，不以 SRT／MD 作為續跑依據，確保 word timestamps 與 canonical transcript 都保留。

## 6. 鍵盤操作

| 按鍵 | 行為 |
| --- | --- |
| `Tab` / `Shift+Tab` | 切換可聚焦欄位 |
| `↑` / `↓` | 移動目前 Tree 游標 |
| `Enter` | 展開／收合資料夾、Dependency、Results 群組；編輯 Diarize speaker 範圍；在媒體上確認後開始或選擇 pipeline 動作 |
| `Space` | Tree 預設展開／收合目前節點 |
| `F5` | 重新掃描 workspace 與快速檢查 |
| `Ctrl+C` | 對進行中的 pipeline 請求取消 |
| `q` / `Ctrl+Q` | 離開 TUI |
| `p` | 開啟獨立的進度框視覺測試 |
| `d` / `t` | 僅在進度框視覺測試中切換顯示階段 |
| `Esc` | 關閉視覺測試、Diarize 設定框、開始確認框或既有輸出動作選擇框 |

## 7. 驗收條件

- 在未安裝 ASR／pyannote 的環境，TUI 仍可快速啟動並以紅色標示缺少依賴；啟動不得載入 PyTorch。
- `Diarize` 可設定並顯示有效的 min/max speaker 範圍；媒體開始前可在確認框核對時長與設定。
- `ffmpeg` 與 Hugging Face access 以獨立項目顯示；token 值絕不出現在畫面或狀態區。
- 完整音檔工作可產生 raw 與 speaker-aligned SRT／MD／JSON／TXT，以及 completed 狀態 JSON。
- `Ctrl+C` 取消時，後續階段不執行；raw 完成時可在下次選擇 Skip to diarize。
- Results 可用 Enter 展開；游標停在 SRT／MD 時下欄顯示內容。
- `Tab` 焦點有高對比外框。
- 取消、失敗、來源不符與成功結果的顏色與操作選項符合本規格。

## 8. 後續工作

1. Setup 的 `config.toml` 互動式編輯與原子儲存。
2. Ollama、Codex CLI、Antigravity CLI 的零成本 backend health check。
3. TXT、JSON、raw artifact 的完整預覽與進階檔案切換。
4. 消除同資料夾不同副檔名共用主檔名的 output collision，而非只以來源指紋警示。
5. 右欄的工作佇列、LLM 後處理、speaker alias 或字幕編輯功能。
