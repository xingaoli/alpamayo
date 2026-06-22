#!/bin/bash
# 清理当前用户占用GPU的进程（含进程树）
# 用法:
#   bash kill_gpu.sh              清理所有占用GPU的进程
#   bash kill_gpu.sh vllm         只清理vllm相关的进程
#   bash kill_gpu.sh -l           仅列出，不清理
#   bash kill_gpu.sh -f           跳过确认直接清理

# 注意：不使用 set -e。本脚本大量依赖 /proc、ps、fuser 等"尽力而为"的探测，
# 被查询的进程随时可能退出，使这些命令返回非零；若开启 errexit，会在读取
# cmdline 时静默中止，连确认提示都不弹出（表现为偶发不弹 yes/no）。
# 改为靠下文的显式判空来容错。
set -uo pipefail

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

# 排除名单：命令行匹配以下任一模式的进程永不被清理
# （例如 sudo nohup 运行的后台 guard 脚本）
EXCLUDE_PATTERNS=(
    "gpu_guard.sh"
)

# 判断某个 PID 是否在排除名单中（按完整命令行子串匹配）
is_excluded_pid() {
    local pid=$1
    local cmdline
    cmdline=$(cat "/proc/$pid/cmdline" 2>/dev/null | tr '\0' ' ')
    [ -z "$cmdline" ] && return 1
    local pat
    for pat in "${EXCLUDE_PATTERNS[@]}"; do
        [[ "$cmdline" == *"$pat"* ]] && return 0
    done
    return 1
}

# 获取当前用户占用GPU的进程PID
get_gpu_pids() {
    nvidia-smi --query-compute-apps=pid,name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null | \
    while IFS=',' read -r pid name mem; do
        pid=$(echo "$pid" | xargs)
        name=$(echo "$name" | xargs)
        mem=$(echo "$mem" | xargs)
        if [ -d "/proc/$pid" ]; then
            proc_user=$(ps -o user= -p "$pid" 2>/dev/null | xargs)
            is_excluded_pid "$pid" && continue
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
    done | tr -s '[:space:]' '\n' | sort -u | while read -r p; do
        is_excluded_pid "$p" || echo "$p"
    done
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
        # 不跨越 IDE 远程会话边界（VSCode / Cursor 等）
        # 这些会话派生的终端 shell argv[0] 仍是普通 bash（如 /usr/bin/bash --init-file .../.vscode-server/...），
        # 无法靠 argv[0] 或 comm 识别，故用完整命令行是否含 .vscode-server / .cursor-server 路径来判断。
        proc_fullcmd=$(cat "/proc/$ppid/cmdline" 2>/dev/null | tr '\0' ' ')
        case "$proc_fullcmd" in
            *.vscode-server/*|*.cursor-server/*) break ;;
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
        if ! is_excluded_pid "$root"; then
            ROOT_PIDS[$root]="$root_cmdline"
            ALL_PIDS[$root]=1
        fi
        for c in $(get_all_children "$root"); do
            is_excluded_pid "$c" || ALL_PIDS[$c]=1
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
        if ! is_excluded_pid "$root"; then
            ROOT_PIDS[$root]="$root_cmdline"
            ALL_PIDS[$root]=1
        fi

        for c in $(get_all_children "$root"); do
            is_excluded_pid "$c" || ALL_PIDS[$c]=1
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
        is_excluded_pid "$c" && continue
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
    is_excluded_pid "$root" && continue
    if kill "$root" 2>/dev/null; then
        echo "终止进程树: PID=$root"
    fi
done

sleep 2

# 检查残留，强制清理
for pid in "${!ALL_PIDS[@]}"; do
    is_excluded_pid "$pid" && continue
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
        is_excluded_pid "$p" && continue
        echo "  PID=$p CMD=$cmdline"
        kill -9 "$p" 2>/dev/null && echo "  已强制终止 PID=$p"
    done
fi

echo ""
echo "清理完成。当前GPU状态:"
nvidia-smi --query-compute-apps=pid,name,used_gpu_memory --format=csv,noheader 2>/dev/null | head -10 || echo "(无GPU占用)"
