#!/bin/bash
# 清理当前用户占用GPU的进程（含进程树）
# 用法:
#   bash kill_gpu.sh              清理所有占用GPU的进程
#   bash kill_gpu.sh vllm         只清理vllm相关的进程
#   bash kill_gpu.sh -l           仅列出，不清理
#   bash kill_gpu.sh -f           跳过确认直接清理

set -euo pipefail

KEYWORD=""
LIST_ONLY=false
FORCE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -l|--list)  LIST_ONLY=true ;;
        -f|--force) FORCE=true ;;
        -*)         echo "未知选项: $1"; exit 1 ;;
        *)          KEYWORD="$1" ;;
    esac
    shift
done

CURRENT_USER=$(whoami)

# 获取当前用户占用GPU的进程PID
get_gpu_pids() {
    nvidia-smi --query-compute-apps=pid,name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null | \
    while IFS=',' read -r pid name mem; do
        pid=$(echo "$pid" | xargs)
        name=$(echo "$name" | xargs)
        mem=$(echo "$mem" | xargs)
        if [ -d "/proc/$pid" ]; then
            proc_user=$(ps -o user= -p "$pid" 2>/dev/null | xargs)
            if [ "$proc_user" = "$CURRENT_USER" ]; then
                if [ -z "$KEYWORD" ] || echo "$name" | grep -qi "$KEYWORD"; then
                    echo "$pid|$name|${mem}MiB"
                fi
            fi
        fi
    done
}

# 通过 fuser 补充查找打开 /dev/nvidia* 设备的进程
get_fuser_pids() {
    for dev in /dev/nvidia*; do
        [ -e "$dev" ] || continue
        [[ "$dev" == /dev/nvidia_uvm* || "$dev" == /dev/nvidia_modeset* || "$dev" == /dev/nvidiactl ]] && continue
        fuser "$dev" 2>/dev/null
    done | tr -s '[:space:]' '\n' | sort -u
}

# 向上追溯进程树，找到顶层父进程（同一个user下）
# 不跨越会话边界：sshd/tmux/screen、登录shell（argv[0]以-开头）
find_root_ancestor() {
    local pid=$1
    local root_pid=$pid
    while true; do
        ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | xargs)
        [ -z "$ppid" ] || [ "$ppid" -eq 0 ] || [ "$ppid" -eq 1 ] && break
        proc_user=$(ps -o user= -p "$ppid" 2>/dev/null | xargs)
        [ "$proc_user" != "$CURRENT_USER" ] && break
        # 不跨越会话边界：sshd、tmux、screen
        proc_comm=$(ps -o comm= -p "$ppid" 2>/dev/null | xargs)
        case "$proc_comm" in
            sshd|sshd:|tmux:|tmux|screen|SCREEN) break ;;
        esac
        # 不跨越登录shell（argv[0]以-开头：-bash, -zsh, -sh等）
        proc_argv0=$(cat "/proc/$ppid/cmdline" 2>/dev/null | tr '\0' '\n' | head -1)
        case "$proc_argv0" in
            -*) break ;;
        esac
        root_pid=$ppid
        pid=$ppid
    done
    echo "$root_pid"
}

# 获取进程树所有子进程（递归）
get_all_children() {
    local ppid=$1
    local children=$(pgrep -P "$ppid" 2>/dev/null || true)
    for c in $children; do
        echo "$c"
        get_all_children "$c"
    done
}

# ---------- 主逻辑 ----------

declare -A ALL_PIDS     # 所有相关PID
declare -A GPU_INFO     # GPU进程信息
declare -A ROOT_PIDS    # 根父进程 -> cmdline

gpu_pids=$(get_gpu_pids)

if [ -z "$gpu_pids" ]; then
    echo "nvidia-smi 未发现GPU占用进程"
    fuser_pids=$(get_fuser_pids)
    if [ -z "$fuser_pids" ]; then
        [ -n "$KEYWORD" ] && echo "(筛选关键字: $KEYWORD)"
        exit 0
    fi
    echo "fuser 检测到以下进程仍持有 /dev/nvidia* 设备，将一并清理:"
    for p in $fuser_pids; do
        [ -d "/proc/$p" ] || continue
        proc_user=$(ps -o user= -p "$p" 2>/dev/null | xargs)
        [ "$proc_user" != "$CURRENT_USER" ] && continue
        cmdline=$(cat "/proc/$p/cmdline" 2>/dev/null | tr '\0' ' ' | cut -c1-80)
        if [ -n "$KEYWORD" ] && ! echo "$cmdline" | grep -qi "$KEYWORD"; then
            continue
        fi
        echo "  PID=$p CMD=$cmdline"
        root=$(find_root_ancestor "$p")
        root_cmdline=$(cat "/proc/$root/cmdline" 2>/dev/null | tr '\0' ' ' | cut -c1-80)
        ROOT_PIDS[$root]="$root_cmdline"
        ALL_PIDS[$root]=1
        for c in $(get_all_children "$root"); do
            ALL_PIDS[$c]=1
        done
    done
    # fuser 路径下可能所有进程都被 KEYWORD 过滤掉了
    if [ ${#ALL_PIDS[@]} -eq 0 ]; then
        [ -n "$KEYWORD" ] && echo "(筛选关键字: $KEYWORD，无匹配进程)"
        exit 0
    fi
fi

if [ -n "$gpu_pids" ]; then
    while IFS='|' read -r pid name mem; do
        GPU_INFO[$pid]="$name | $mem"

        # 向上找根父进程
        root=$(find_root_ancestor "$pid")

        # 收集根父进程及其所有子进程
        root_cmdline=$(cat "/proc/$root/cmdline" 2>/dev/null | tr '\0' ' ' | cut -c1-80)
        ROOT_PIDS[$root]="$root_cmdline"

        ALL_PIDS[$root]=1
        for c in $(get_all_children "$root"); do
            ALL_PIDS[$c]=1
        done
    done <<< "$gpu_pids"
fi

# 展示信息
if [ -n "$gpu_pids" ]; then
    echo "=== 占用GPU的进程 ==="
    printf "%-10s %-30s %s\n" "PID" "NAME" "GPU MEM"
    echo "--------------------------------------------------------------"
    while IFS='|' read -r pid name mem; do
        printf "%-10s %-30s %s\n" "$pid" "$name" "$mem"
    done <<< "$gpu_pids"
    echo ""
fi

echo "=== 关联的进程树（将被一并清理）==="
for root in "${!ROOT_PIDS[@]}"; do
    echo "  根进程: PID=$root  ${ROOT_PIDS[$root]}"
    for c in $(get_all_children "$root" | sort -n); do
        c_cmd=$(cat "/proc/$c/cmdline" 2>/dev/null | tr '\0' ' ' | cut -c1-60)
        gpu_mark=""
        [ -n "${GPU_INFO[$c]+_}" ] && gpu_mark=" <-- GPU"
        echo "    └─ PID=$c  $c_cmd$gpu_mark"
    done
done
echo ""

total=${#ALL_PIDS[@]}
echo "共 $total 个进程将被清理"

if $LIST_ONLY; then
    exit 0
fi

if ! $FORCE; then
    read -rp "确认清理? [y/N] " confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        echo "已取消"
        exit 0
    fi
fi

# 从根进程开始杀整棵树
for root in "${!ROOT_PIDS[@]}"; do
    if kill "$root" 2>/dev/null; then
        echo "终止进程树: PID=$root"
    fi
done

sleep 2

# 检查残留，强制清理
for pid in "${!ALL_PIDS[@]}"; do
    if [ -d "/proc/$pid" ]; then
        kill -9 "$pid" 2>/dev/null && echo "强制终止: PID=$pid"
    fi
done

sleep 1

# 用 fuser 做最终检查
fuser_pids=$(get_fuser_pids)
if [ -n "$fuser_pids" ]; then
    echo ""
    echo "fuser 检测到以下进程仍持有 GPU 设备:"
    for p in $fuser_pids; do
        [ -d "/proc/$p" ] || continue
        proc_user=$(ps -o user= -p "$p" 2>/dev/null | xargs)
        [ "$proc_user" != "$CURRENT_USER" ] && continue
        cmdline=$(cat "/proc/$p/cmdline" 2>/dev/null | tr '\0' ' ' | cut -c1-80)
        echo "  PID=$p CMD=$cmdline"
        kill -9 "$p" 2>/dev/null && echo "  已强制终止 PID=$p"
    done
fi

echo ""
echo "清理完成。当前GPU状态:"
nvidia-smi --query-compute-apps=pid,name,used_gpu_memory --format=csv,noheader 2>/dev/null | head -10 || echo "(无GPU占用)"
