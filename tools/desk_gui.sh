#!/bin/bash
# 离开工位：关掉图形桌面，保留 tmux 里的训练；回来再开桌面。
#
# 用法:
#   ./tools/desk_gui.sh leave          # 检查 tmux 后切到多用户（无 GUI）
#   ./tools/desk_gui.sh leave --force  # 不检查 tmux，强制关 GUI
#   ./tools/desk_gui.sh back           # 恢复图形界面
#   ./tools/desk_gui.sh status         # 当前是 graphical 还是 multi-user
#
# 训练必须先在 tmux 里跑，例如:
#   tmux new -s train
#   # ... bash run_2026_journal_experiments.sh ...
#   # Ctrl-b d 脱离后再执行 leave

set -euo pipefail

usage() {
    sed -n '2,14p' "$0" | sed 's/^# \?//'
    exit "${1:-0}"
}

need_sudo() {
    if [[ "$(id -u)" -eq 0 ]]; then
        "$@"
    else
        sudo "$@"
    fi
}

cmd_status() {
    if systemctl is-active --quiet graphical.target 2>/dev/null; then
        echo "graphical.target: active (桌面开着)"
    else
        echo "graphical.target: inactive"
    fi
    if systemctl is-active --quiet multi-user.target 2>/dev/null; then
        echo "multi-user.target: active"
    fi
    echo "default: $(systemctl get-default 2>/dev/null || true)"
    if command -v tmux >/dev/null; then
        echo "tmux sessions:"
        tmux ls 2>/dev/null || echo "  (none)"
    fi
    if command -v nvidia-smi >/dev/null; then
        nvidia-smi --query-gpu=persistence_mode,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null \
            | sed 's/^/gpu: /' || true
    fi
}

cmd_leave() {
    local force=false
    [[ "${1:-}" == --force ]] && force=true

    if [[ "${force}" != true ]]; then
        if ! command -v tmux >/dev/null || ! tmux ls >/dev/null 2>&1; then
            echo "错误: 没有 tmux 会话。训练若不在 tmux 里，关 GUI 会一起杀掉。" >&2
            echo "先: tmux new -s train   再在里面开训；或: $0 leave --force" >&2
            exit 1
        fi
        echo "将保留以下 tmux 会话:"
        tmux ls
    else
        echo "警告: --force，不检查 tmux"
    fi

    # 可选：打开 persistence（失败不影响关 GUI）
    if command -v nvidia-smi >/dev/null; then
        need_sudo nvidia-smi -pm 1 >/dev/null 2>&1 || true
    fi

    echo "切换到 multi-user.target（关闭 GUI）…"
    need_sudo systemctl isolate multi-user.target
    echo "完成。回来后在 TTY 执行: $(cd "$(dirname "$0")" && pwd)/desk_gui.sh back"
    echo "或: sudo systemctl isolate graphical.target"
}

cmd_back() {
    echo "恢复 graphical.target…"
    need_sudo systemctl isolate graphical.target
    echo "完成。若黑屏，试 Ctrl+Alt+F1 / F2。"
}

case "${1:-}" in
    leave) shift; cmd_leave "$@" ;;
    back) cmd_back ;;
    status) cmd_status ;;
    -h|--help|help|"") usage 0 ;;
    *) echo "未知命令: $1" >&2; usage 1 ;;
esac
