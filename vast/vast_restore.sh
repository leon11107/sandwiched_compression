#!/usr/bin/env bash
# 新 Vast 實例：把整個 sandwich 目錄 + Claude 對話紀錄從 B2 拉回來
# 用法：bash vast_restore.sh <B2_KEY_ID> <B2_APP_KEY> <BUCKET> <LOCAL_DIR> <REMOTE_PATH>
# 例：
#   bash vast_restore.sh 0051abc K001xyz my-bucket /workspace/sandwiched_compression sandwich-main

set -euo pipefail

B2_KEY_ID="${1:?B2 keyID required}"
B2_APP_KEY="${2:?B2 applicationKey required}"
BUCKET="${3:?bucket name required}"
LOCAL_DIR="${4:?local directory required}"
REMOTE_PATH="${5:?remote path required}"

LOCAL_DIR="${LOCAL_DIR%/}"
REMOTE_PATH="${REMOTE_PATH%/}"

REPO_REMOTE="b2:${BUCKET}/${REMOTE_PATH}/repo"
CLAUDE_REMOTE="b2:${BUCKET}/${REMOTE_PATH}/claude_projects"

if ! command -v rclone >/dev/null 2>&1; then
  curl -fsSL https://rclone.org/install.sh | bash
fi

mkdir -p "${HOME}/.config/rclone"
cat > "${HOME}/.config/rclone/rclone.conf" <<EOF
[b2]
type = b2
account = ${B2_KEY_ID}
key = ${B2_APP_KEY}
EOF
chmod 600 "${HOME}/.config/rclone/rclone.conf"

mkdir -p "${LOCAL_DIR}"
echo "[*] Pulling ${REPO_REMOTE} -> ${LOCAL_DIR}..."
rclone sync "${REPO_REMOTE}/" "${LOCAL_DIR}/" \
  --transfers 8 --checkers 16 --progress

# Claude Code 對話紀錄（若雲端有存）
if rclone lsd "${CLAUDE_REMOTE}" >/dev/null 2>&1; then
  mkdir -p "${HOME}/.claude/projects"
  echo "[*] Pulling ${CLAUDE_REMOTE} -> ~/.claude/projects/..."
  rclone sync "${CLAUDE_REMOTE}/" "${HOME}/.claude/projects/" \
    --transfers 4 --progress
  echo "[+] Claude conversation history restored to ~/.claude/projects/"
fi

echo ""
echo "[+] Restore complete."
echo "    Local dir : ${LOCAL_DIR}"
echo ""
echo "Next steps:"
echo "  1) bash ${LOCAL_DIR}/vast/vast_backup_setup.sh <KEY_ID> <APP_KEY> ${BUCKET} ${LOCAL_DIR} ${REMOTE_PATH}"
echo "     # 在這台新機器重新啟動自動 sync"
echo "  2) cd ${LOCAL_DIR} && python <your-train-script>.py --resume   # 接續訓練"
