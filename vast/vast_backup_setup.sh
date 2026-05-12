#!/usr/bin/env bash
# Vast.ai 端：安裝 rclone + 設定 B2 + 每 N 分鐘 sync 整個目錄到 Backblaze B2
# 用法：
#   bash vast_backup_setup.sh <B2_KEY_ID> <B2_APP_KEY> <BUCKET> <LOCAL_DIR> <REMOTE_PATH> [interval_min]
# 例：
#   bash vast_backup_setup.sh 0051abc K001xyz my-bucket /workspace/sandwiched_compression sandwich-main 5

set -euo pipefail

B2_KEY_ID="${1:?B2 keyID required}"
B2_APP_KEY="${2:?B2 applicationKey required}"
BUCKET="${3:?bucket name required}"
LOCAL_DIR="${4:?local directory required}"
REMOTE_PATH="${5:?remote path required}"
INTERVAL_MIN="${6:-10}"

LOCAL_DIR="${LOCAL_DIR%/}"
REMOTE_PATH="${REMOTE_PATH%/}"

REPO_REMOTE="b2:${BUCKET}/${REMOTE_PATH}/repo"
CLAUDE_REMOTE="b2:${BUCKET}/${REMOTE_PATH}/claude_projects"

# 1. rclone
if ! command -v rclone >/dev/null 2>&1; then
  echo "[*] Installing rclone..."
  curl -fsSL https://rclone.org/install.sh | bash
fi

# 2. rclone 設定
mkdir -p "${HOME}/.config/rclone"
cat > "${HOME}/.config/rclone/rclone.conf" <<EOF
[b2]
type = b2
account = ${B2_KEY_ID}
key = ${B2_APP_KEY}
hard_delete = false
EOF
chmod 600 "${HOME}/.config/rclone/rclone.conf"

# 3. 確認本地資料夾存在
if [[ ! -d "${LOCAL_DIR}" ]]; then
  echo "[!] ${LOCAL_DIR} not found."
  exit 1
fi

# 4. 測試 B2
rclone lsd "b2:${BUCKET}" >/dev/null && echo "[+] B2 OK"

# 5. sync 腳本：整個 sandwich 目錄 + Claude Code 對話紀錄（原封不動，不加任何 exclude）
cat > /usr/local/bin/vast-sync.sh <<EOF
#!/usr/bin/env bash
set -e
# 5a. 主目錄
rclone sync "${LOCAL_DIR}" "${REPO_REMOTE}/" \\
  --transfers 8 --checkers 16 \\
  --log-file=/var/log/vast-sync.log --log-level INFO

# 5b. Claude Code 對話紀錄
if [[ -d "\${HOME}/.claude/projects" ]]; then
  rclone sync "\${HOME}/.claude/projects" "${CLAUDE_REMOTE}/" \\
    --transfers 4 --log-file=/var/log/vast-sync.log --log-level INFO || true
fi
EOF
chmod +x /usr/local/bin/vast-sync.sh

# 6. cron
if ! command -v cron >/dev/null 2>&1; then
  apt-get update && apt-get install -y cron
fi
service cron start || cron

(crontab -l 2>/dev/null | grep -v vast-sync.sh; \
 echo "*/${INTERVAL_MIN} * * * * /usr/local/bin/vast-sync.sh") | crontab -

# 7. 立刻跑一次
/usr/local/bin/vast-sync.sh

echo ""
echo "[+] Backup configured."
echo "    Local           : ${LOCAL_DIR}"
echo "    Remote (repo)   : ${REPO_REMOTE}"
echo "    Remote (claude) : ${CLAUDE_REMOTE}"
echo "    Interval        : every ${INTERVAL_MIN} min"
echo "    Manual sync     : /usr/local/bin/vast-sync.sh"
echo "    Log             : /var/log/vast-sync.log"
