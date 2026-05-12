# Vast.ai 實驗備份 / 續傳工具

在 [Vast.ai](https://vast.ai/) 上跑訓練時，把整個 sandwich 目錄（程式碼、checkpoints、log、實驗結果、Claude 對話紀錄）增量備份到 Backblaze B2，實例被搶佔或主動換機器時可以**無痛接續**。

## 為什麼需要這個

Vast.ai 的 GPU 實例（尤其 interruptible 機種）隨時可能消失，本地磁碟一起沒。沒有自動備份的話：

- 訓練到一半的 checkpoint 全部消失
- 實驗筆記、訓練 log 全部消失
- 想換更便宜/更快的機器時要手動搬一堆檔

這套工具用 `rclone` + `cron` 把指定的本地目錄（例如整個 sandwich repo）每 N 分鐘**完整鏡像**到 B2；換機器時一行指令拉回來，包含 `~/.claude/projects/` 的 Claude Code 對話紀錄。

**重點：完全不加 exclude，所有檔案原封不動同步。**

## 為什麼選 Backblaze B2

| | 儲存 (USD/GB/月) | 下載 (USD/GB) |
|---|---|---|
| AWS S3 | 0.023 | 0.09 |
| **Backblaze B2** | **0.006** | **0.01** |

對 checkpoint 場景（寫多讀少）B2 大概便宜 S3 一個量級。同時 rclone 完全相容。

---

## 前置作業（一次性）

1. 註冊 [Backblaze 帳號](https://www.backblaze.com/)
2. 建立 bucket（Private 即可）
3. 建立 Application Key，記下：
   - `keyID`（看起來像 `0051abc...`）
   - `applicationKey`（看起來像 `K001xyz...`）

---

## 兩個腳本

### 1. `vast_backup_setup.sh` — 啟動自動備份

```bash
bash vast/vast_backup_setup.sh <KEY_ID> <APP_KEY> <BUCKET> <LOCAL_DIR> <REMOTE_PATH> [interval_min]

# 例：每 5 分鐘 sync 整個 sandwich 目錄
bash vast/vast_backup_setup.sh 0051abc K001xyz my-bucket \
  /workspace/sandwiched_compression sandwich-main 5
```

參數說明：

| 參數 | 說明 | 範例 |
|---|---|---|
| `KEY_ID` / `APP_KEY` | B2 Application Key 的兩個欄位 | `0051abc...` / `K001xyz...` |
| `BUCKET` | B2 bucket 名稱 | `my-bucket` |
| `LOCAL_DIR` | 本地要備份的目錄（會整個鏡像） | `/workspace/sandwiched_compression` |
| `REMOTE_PATH` | 在 bucket 裡的子路徑前綴（自己取名） | `sandwich-main` |
| `interval_min` | cron 間隔分鐘數（預設 10） | `5` |

做的事：

1. 裝 `rclone`（若未裝）
2. 寫入 B2 設定到 `~/.config/rclone/rclone.conf`
3. 註冊 cron job 每 N 分鐘執行：
   - `rclone sync <LOCAL_DIR>/ → b2:<BUCKET>/<REMOTE_PATH>/repo/`
   - `rclone sync ~/.claude/projects/ → b2:<BUCKET>/<REMOTE_PATH>/claude_projects/`
4. 立刻跑一次首發 sync

雲端結構：

```
b2:<BUCKET>/<REMOTE_PATH>/
├── repo/              ← 整個 LOCAL_DIR 鏡像（含 .git、checkpoints、experiments...）
└── claude_projects/   ← ~/.claude/projects/ 鏡像
```

查看備份狀態：
```bash
tail -f /var/log/vast-sync.log
crontab -l                       # 看 cron 排程
/usr/local/bin/vast-sync.sh      # 手動觸發一次
```

### 2. `vast_restore.sh` — 換機器時把整包拉回來

```bash
bash vast/vast_restore.sh <KEY_ID> <APP_KEY> <BUCKET> <LOCAL_DIR> <REMOTE_PATH>

# 例：
bash vast/vast_restore.sh 0051abc K001xyz my-bucket \
  /workspace/sandwiched_compression sandwich-main
```

把雲端 `b2:<BUCKET>/<REMOTE_PATH>/repo/` 整包拉回 `<LOCAL_DIR>`，包含 `.git/` / code / checkpoints / experiments / logs / 任何你存在裡面的東西。Claude Code 對話紀錄也會還原到 `~/.claude/projects/`。

---

## 完整工作流程

### 第一次開始實驗

```bash
# 1. clone repo（或在 Vast 實例上 git pull 你自己的 fork）
git clone <this-repo> /workspace/sandwiched_compression
cd /workspace/sandwiched_compression

# 2. 啟動自動備份（每 5 分鐘）
bash vast/vast_backup_setup.sh <KEY_ID> <APP_KEY> <BUCKET> \
  /workspace/sandwiched_compression sandwich-main 5

# 3. 開跑
python <your-train-script>.py
```

### 實例掛掉 / 換機器

```bash
# 1. ssh 進新機器，先 clone repo（目的只是為了拿到 vast/vast_restore.sh 這個腳本）
git clone <this-repo> /workspace/sandwiched_compression
cd /workspace/sandwiched_compression

# 2. 從 B2 把整包拉回來（會覆蓋剛 clone 的內容，包含 .git 都換成備份版本）
bash vast/vast_restore.sh <KEY_ID> <APP_KEY> <BUCKET> \
  /workspace/sandwiched_compression sandwich-main

# 3. 重啟自動備份（每台新機器都要做一次）
bash vast/vast_backup_setup.sh <KEY_ID> <APP_KEY> <BUCKET> \
  /workspace/sandwiched_compression sandwich-main 5

# 4. 接續訓練（你的訓練腳本要支援 --resume，自己讀最新 checkpoint）
python <your-train-script>.py --resume
```

> 注意：步驟 1 的 git clone 只是 bootstrap 用來拿到 `vast_restore.sh`。clone 完馬上會被步驟 2 的 restore 覆蓋掉，所以 clone 的內容無所謂。

---

## 訓練腳本需要配合的設計

腳本本身不限定訓練框架，但要讓「換機器接續」真的有用，你的訓練腳本至少要：

1. **把所有產出寫進 `<LOCAL_DIR>` 底下**，不要寫到別處（否則備份範圍涵蓋不到）
2. **存 checkpoint 時包含完整狀態**：
   - 模型權重、optimizer state
   - epoch / step counter
   - best metric so far
   - RNG state（Python / NumPy / TF/PyTorch）
   - dataset iterator 位置（想中途接續才需要）
3. **支援 `--resume`**：啟動時若發現 checkpoint 目錄有東西就還原
4. **stdout 同時寫入檔案**：例如用 `tee` 或在程式碼裡寫 `sys.stdout = Tee(sys.stdout, open('stdout.log', 'a'))`

---

## 進階：開機自動續傳

把 restore + backup_setup 寫進 Vast 實例的 `onstart.sh`，租新機器後**完全不用 ssh** 就會自動接續：

```bash
# 在 Vast 建立實例時，把下面這段貼到 On-start Script
#!/bin/bash
git clone <this-repo> /workspace/sandwiched_compression
cd /workspace/sandwiched_compression
bash vast/vast_restore.sh "$B2_KEY_ID" "$B2_APP_KEY" "$BUCKET" \
  /workspace/sandwiched_compression "$REMOTE_PATH"
bash vast/vast_backup_setup.sh "$B2_KEY_ID" "$B2_APP_KEY" "$BUCKET" \
  /workspace/sandwiched_compression "$REMOTE_PATH" 5
python <your-train-script>.py --resume > stdout.log 2>&1 &
```

把 `B2_KEY_ID` / `B2_APP_KEY` / `BUCKET` / `REMOTE_PATH` 用 Vast 的環境變數設定，避免寫死在腳本裡。

---

## 安全性

- **B2 credentials 不要 commit 進 repo**。腳本設計成從命令列參數傳入，不會留在程式碼裡。
- 若用 onstart.sh 自動化，把 credentials 放在 Vast 環境變數而不是腳本本體。
- B2 Application Key 建議只給單一 bucket 的權限，不要用 master key。

## 故障排除

| 症狀 | 檢查 |
|---|---|
| `rclone: command not found` | setup 腳本會自動裝；手動裝：`curl https://rclone.org/install.sh \| bash` |
| sync 沒在跑 | `service cron status` / `crontab -l` 看 cron 是否活著 |
| 找不到 checkpoint | `rclone ls b2:<BUCKET>/<REMOTE_PATH>/repo/` 看雲端有沒有 |
| sync 一直失敗 | `tail -100 /var/log/vast-sync.log` 看 rclone 錯誤訊息 |
