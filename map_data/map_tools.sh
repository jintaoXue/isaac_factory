#!/usr/bin/env bash
#
# 一键运行本目录下的地图 / 路网相关脚本，方便测试与可视化。
#
# 用法示例（在仓库根目录执行）：
#   bash map_data/map_tools.sh generate-points-human
#   bash map_data/map_tools.sh add-points-human
#   bash map_data/map_tools.sh gen-paths
#   bash map_data/map_tools.sh view-paths
#   bash map_data/map_tools.sh overlay-points

# bash map_data/map_tools.sh edit-map-hall
# bash map_data/map_tools.sh edit-map4robot
# bash map_data/map_tools.sh edit-points-robot
# bash map_data/map_tools.sh gen-paths-robot
# bash map_data/map_tools.sh view-paths-robot

# 所有子命令都可以在后面追加额外参数，这些参数会原样传给对应的 python 脚本：
#   bash map_data/map_tools.sh gen-paths --interp-step 3.0
#

set -euo pipefail

# 当前脚本所在的 map_data 目录
# 使用 $0，而不是 BASH_SOURCE，以避免被其他封装脚本/alias 干扰。
MAP_DIR="$(cd "$(dirname "$0")" && pwd)"

# 使用当前环境中的 python3（由你自己在 shell/conda 里控制）
PYTHON_BIN="python3"

# 可以根据需要自行修改下面这几个默认路径/参数（人行）
MAP_IMG_HUMAN="${MAP_DIR}/occupancy_map4human.png"
POINTS_JSON_HUMAN="${MAP_DIR}/map_points_human.json"
PATHS_JSON_HUMAN="${MAP_DIR}/map_paths_human.json"
OVERLAY_IMG_WITH_POINTS_HUMAN="${MAP_DIR}/map_with_points_human.png"
DEFAULT_INTERP_STEP="5.0"

# 机器人 AGV 路径编辑默认路径
MAP_IMG_ROBOT="${MAP_DIR}/occupancy_map4robot.png"
POINTS_JSON_ROBOT="${MAP_DIR}/map_points_robot.json"

cmd="${1:-}"
shift || true

case "${cmd}" in
  generate-points-human)
    # 从零开始生成路网点（不加载旧点）
    # 为避免误覆盖：如果已存在 points 文件，先自动备份一份再开始新建
    if [[ -f "${POINTS_JSON_HUMAN}" ]]; then
      ts="$(date +%Y%m%d_%H%M%S)"
      cp -f "${POINTS_JSON_HUMAN}" "${POINTS_JSON_HUMAN}.bak.${ts}"
      echo "[INFO] 已备份旧点文件到: ${POINTS_JSON_HUMAN}.bak.${ts}" >&2
    fi
    "${PYTHON_BIN}" "${MAP_DIR}/map_points_generation.py" \
      --map "${MAP_IMG_HUMAN}" \
      --out "${POINTS_JSON_HUMAN}" \
      --overlay-out "${OVERLAY_IMG_WITH_POINTS_HUMAN}" \
      "$@"
    ;;

  add-points-human)
    # 交互式编辑路网点：
    # - 若已存在 points 文件，则先加载后再允许“添加/删除”
    # - 保存仍覆盖写回到同一个 points 文件
    load_args=()
    if [[ -f "${POINTS_JSON_HUMAN}" ]]; then
      load_args+=( --load "${POINTS_JSON_HUMAN}" )
    fi
    "${PYTHON_BIN}" "${MAP_DIR}/map_points_generation.py" \
      --map "${MAP_IMG_HUMAN}" \
      --out "${POINTS_JSON_HUMAN}" \
      "${load_args[@]}" \
      --overlay-out "${OVERLAY_IMG_WITH_POINTS_HUMAN}" \
      "$@"
    ;;

  edit-points-robot)
    # 交互式编辑/标注机器人 AGV 路网点（使用 hall 占据图）
    "${PYTHON_BIN}" "${MAP_DIR}/map_points_generation.py" \
      --map "${MAP_IMG_ROBOT}" \
      --out "${POINTS_JSON_ROBOT}" \
      "$@"
    ;;

  edit-map-hall)
    # 交互式编辑 hall 占据图（画笔/橡皮擦：白=可通行，黑=占据）
    # 默认保存为 occupancy_map4robot.png（可通过 --out 手动覆盖）
    "${PYTHON_BIN}" "${MAP_DIR}/map_image_editor.py" \
      --map "${MAP_IMG_ROBOT}" \
      --out "${MAP_DIR}/occupancy_map4robot.png" \
      "$@"
    ;;

  edit-map4robot)
    # 直接编辑 occupancy_map4robot.png 本身（细调/修补现有机器人占据图）
    "${PYTHON_BIN}" "${MAP_DIR}/map_image_editor.py" \
      --map "${MAP_DIR}/occupancy_map4robot.png" \
      --out "${MAP_DIR}/occupancy_map4robot.png" \
      "$@"
    ;;

  edit-map)
    # 交互式编辑任意占据图（用法：bash map_data/map_tools.sh edit-map <png路径> [--out ...]）
    map_in="${1:-}"
    if [[ -z "${map_in}" ]]; then
      echo "用法: bash map_data/map_tools.sh edit-map <png路径> [额外参数...]" >&2
      exit 2
    fi
    shift || true
    "${PYTHON_BIN}" "${MAP_DIR}/map_image_editor.py" \
      --map "${map_in}" \
      "$@"
    ;;

  gen-paths)
    # 基于占据图 + 路网点，预计算所有点对最短路，并保存插值后的 [x,y,yaw] samples
    "${PYTHON_BIN}" "${MAP_DIR}/map_paths_generation.py" \
      --map "${MAP_IMG_HUMAN}" \
      --points "${POINTS_JSON_HUMAN}" \
      --out "${PATHS_JSON_HUMAN}" \
      --interp-step "${DEFAULT_INTERP_STEP}" \
      "$@"
    ;;

  view-paths)
    # 打开查看器：显示地图 + 所有点编号，终端输入起终点 id 高亮路径并打印插值时间
    "${PYTHON_BIN}" "${MAP_DIR}/map_paths_viewer.py" \
      --paths "${PATHS_JSON_HUMAN}" \
      "$@"
    ;;

  overlay-points)
    # 将当前路网点叠加到占据图上，生成一张带点的地图图片
    "${PYTHON_BIN}" "${MAP_DIR}/map_points_generation.py" \
      --map "${MAP_IMG_HUMAN}" \
      --out "${POINTS_JSON_HUMAN}" \
      --overlay-map "${MAP_IMG_HUMAN}" \
      --overlay-points "${POINTS_JSON_HUMAN}" \
      --overlay-out "${OVERLAY_IMG_WITH_POINTS_HUMAN}" \
      "$@"
    ;;

  ""|-h|--help|help)
    cat <<EOF
用法: bash map_data/map_tools.sh <子命令> [额外参数...]

子命令：
  generate-points-human
                  从零开始新建路网点（不加载旧点）；若已存在点文件会自动备份为 .bak.<时间戳>
  add-points-human
                  交互式编辑/标注路网点，保存到 ${POINTS_JSON_HUMAN}，使用 ${MAP_IMG_HUMAN}
  edit-points-robot
                  交互式编辑/标注机器人 AGV 路网点，保存到 ${POINTS_JSON_ROBOT}，使用 ${MAP_IMG_ROBOT}
  edit-map-hall   交互式编辑 hall 占据图（画笔/橡皮擦：白=可通行，黑=占据），默认保存为 occupancy_map4robot.png
  edit-map4robot  在现有 occupancy_map4robot.png 上直接编辑（细调/修补），默认覆盖该文件
  edit-map        交互式编辑任意占据图（画笔/橡皮擦：白=可通行，黑=占据）
  gen-paths       预计算所有点对最短路并保存到 ${PATHS_JSON_HUMAN}，同时生成插值后的 [x,y,yaw] samples
  view-paths      打开查看器，显示地图和所有点编号，在终端输入起终点 id 高亮路径并打印插值耗时
  overlay-points  将路网点叠加到占据图上，生成 ${OVERLAY_IMG_WITH_POINTS_HUMAN}

所有子命令都支持在后面追加原生 python 参数，例如：
  bash map_data/map_tools.sh gen-paths --interp-step 3.0
  bash map_data/map_tools.sh gen-paths --k-neighbors 12 --max-neighbor-dist 300
EOF
    ;;

  *)
    echo "未知子命令: ${cmd}" >&2
    echo "请使用: bash map_data/map_tools.sh --help 查看帮助" >&2
    exit 1
    ;;
esac

