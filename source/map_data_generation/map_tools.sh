#!/usr/bin/env bash
#
# 一键运行本目录下的地图 / 路网相关脚本，方便测试与可视化。
#
# 用法示例（在仓库根目录执行）：
#   bash source/map_data_generation/map_tools.sh generate-points-human
#   bash source/map_data_generation/map_tools.sh add-points-human
#   bash source/map_data_generation/map_tools.sh gen-routes
#   bash source/map_data_generation/map_tools.sh view-routes
#   bash source/map_data_generation/map_tools.sh overlay-points

# bash source/map_data_generation/map_tools.sh edit-map4robot
# bash source/map_data_generation/map_tools.sh generate-points-robot
# bash source/map_data_generation/map_tools.sh add-points-robot
# bash source/map_data_generation/map_tools.sh gen-routes-robot
# bash source/map_data_generation/map_tools.sh view-routes-robot

# 坐标批量转换（用于“旧版像素 JSON”迁移到 Isaac Sim 坐标系）：
# - 读取 config.json 中的 points_path / routes_path（以及 *_robot）
# - 将 JSON 内的 (x,y) 从 PNG 像素坐标转换为 Isaac Sim 坐标，并写入 coordinate_frame=isaac_sim
# 示例（仓库根目录执行）：
#   bash source/map_data_generation/map_tools.sh trans-json-coordinates source/map_data_generation/config.json --in-place
# 所有子命令都可以在后面追加额外参数，这些参数会原样传给对应的 python 脚本：
#   bash source/map_data_generation/map_tools.sh gen-routes --interp-step 3.0
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
ROUTES_JSON_HUMAN="~/work/Dataset/HC_data/map_data/map_routes_human.json"
OVERLAY_IMG_WITH_POINTS_HUMAN="${MAP_DIR}/map_with_points_human.png"
DEFAULT_INTERP_STEP="5.0"

# 机器人 AGV 路径编辑默认路径
MAP_IMG_ROBOT="${MAP_DIR}/occupancy_map4robot.png"
POINTS_JSON_ROBOT="${MAP_DIR}/map_points_robot.json"
ROUTES_JSON_ROBOT="~/work/Dataset/HC_data/map_data/map_routes_robot.json"
OVERLAY_IMG_WITH_POINTS_ROBOT="${MAP_DIR}/map_with_points_robot.png"

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

  generate-points-robot)
    # 交互式编辑/标注机器人 AGV 路网点（使用 hall 占据图）
    "${PYTHON_BIN}" "${MAP_DIR}/map_points_generation.py" \
      --map "${MAP_IMG_ROBOT}" \
      --out "${POINTS_JSON_ROBOT}" \
      "$@"
    ;;

  add-points-robot)
    # 交互式编辑机器人路网点（添加/删除）：
    # - 若已存在 points 文件，则先加载后再允许“添加/删除”
    # - 保存仍覆盖写回到同一个 points 文件
    load_args=()
    if [[ -f "${POINTS_JSON_ROBOT}" ]]; then
      load_args+=( --load "${POINTS_JSON_ROBOT}" )
    fi
    "${PYTHON_BIN}" "${MAP_DIR}/map_points_generation.py" \
      --map "${MAP_IMG_ROBOT}" \
      --out "${POINTS_JSON_ROBOT}" \
      "${load_args[@]}" \
      --overlay-out "${OVERLAY_IMG_WITH_POINTS_ROBOT}" \
      "$@"
    ;;

  edit-map4robot)
    # 直接编辑 occupancy_map4robot.png 本身（细调/修补现有机器人占据图）
    "${PYTHON_BIN}" "${MAP_DIR}/map_image_editor.py" \
      --map "${MAP_DIR}/occupancy_map4robot_raw.png" \
      --out "${MAP_DIR}/occupancy_map4robot.png" \
      "$@"
    ;;

  gen-routes)
    # 基于占据图 + 路网点，预计算所有点对最短路，并保存插值后的 [x,y,yaw] samples
    "${PYTHON_BIN}" "${MAP_DIR}/map_paths_generation.py" \
      --map "${MAP_IMG_HUMAN}" \
      --points "${POINTS_JSON_HUMAN}" \
      --out "${ROUTES_JSON_HUMAN}" \
      --interp-step "${DEFAULT_INTERP_STEP}" \
      "$@"
    ;;

  gen-routes-robot)
    # 基于机器人占据图 + 机器人路网点，预计算所有点对最短路（routes）
    "${PYTHON_BIN}" "${MAP_DIR}/map_paths_generation.py" \
      --map "${MAP_IMG_ROBOT}" \
      --points "${POINTS_JSON_ROBOT}" \
      --out "${ROUTES_JSON_ROBOT}" \
      --interp-step "${DEFAULT_INTERP_STEP}" \
      "$@"
    ;;

  view-routes)
    # 打开查看器：显示地图 + 所有点编号，终端输入起终点 id 高亮路径并打印插值时间
    "${PYTHON_BIN}" "${MAP_DIR}/map_paths_viewer.py" \
      --config "${MAP_DIR}/config.json" \
      "$@"
    ;;

  view-routes-robot)
    # 打开机器人 routes 查看器：显式指定 robot 的 routes/map/points（避免与 human config 混用）
    "${PYTHON_BIN}" "${MAP_DIR}/map_paths_viewer.py" \
      --config "${MAP_DIR}/config.json" \
      --paths "${ROUTES_JSON_ROBOT}" \
      --map "${MAP_IMG_ROBOT}" \
      --points "${POINTS_JSON_ROBOT}" \
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

  trans-json-coordinates)
    # 批量将 points/routes JSON 从 PNG 像素坐标转换为 Isaac Sim 坐标（基于 config.json 角点映射）
    # 用法示例：
    #   bash source/map_data_generation/map_tools.sh trans-json-coordinates source/map_data_generation/config.json --in-place
    #   bash source/map_data_generation/map_tools.sh trans-json-coordinates source/map_data_generation/config.json --out-dir source/map_data_generation/converted
    cfg_path="${1:-}"
    if [[ -z "${cfg_path}" ]]; then
      echo "缺少 config.json 路径。用法：" >&2
      echo "  bash source/map_data_generation/map_tools.sh trans-json-coordinates source/map_data_generation/config.json [--in-place | --out-dir DIR]" >&2
      exit 2
    fi
    shift || true
    "${PYTHON_BIN}" "${MAP_DIR}/trans_json_coordinates.py" \
      --config "${cfg_path}" \
      "$@"
    ;;

  ""|-h|--help|help)
    cat <<EOF
用法: bash source/map_data_generation/map_tools.sh <子命令> [额外参数...]

子命令：
  generate-points-human
                  从零开始新建路网点（不加载旧点）；若已存在点文件会自动备份为 .bak.<时间戳>
  add-points-human
                  交互式编辑/标注路网点，保存到 ${POINTS_JSON_HUMAN}，使用 ${MAP_IMG_HUMAN}
  generate-points-robot
                  交互式生成机器人 AGV 路网点，保存到 ${POINTS_JSON_ROBOT}，使用 ${MAP_IMG_ROBOT}
  add-points-robot
                  交互式编辑机器人路网点（添加/删除/画框撒点），保存到 ${POINTS_JSON_ROBOT}，使用 ${MAP_IMG_ROBOT}
  edit-map-hall   交互式编辑 hall 占据图（画笔/橡皮擦：白=可通行，黑=占据），默认保存为 occupancy_map4robot.png
  edit-map4robot  在现有 occupancy_map4robot.png 上直接编辑（细调/修补），默认覆盖该文件
  edit-map        交互式编辑任意占据图（画笔/橡皮擦：白=可通行，黑=占据）
  gen-routes      预计算人行 routes 并保存到 ${ROUTES_JSON_HUMAN}（含 [x,y,yaw] samples）
  gen-routes-robot 预计算机器人 routes 并保存到 ${ROUTES_JSON_ROBOT}（含 [x,y,yaw] samples）
  view-routes     打开查看器（默认读 config.json），显示地图和所有点编号，在终端输入起终点 id 高亮路径并打印插值耗时
  view-routes-robot 打开机器人 routes 查看器（默认使用 ${ROUTES_JSON_ROBOT} / ${POINTS_JSON_ROBOT} / ${MAP_IMG_ROBOT}）
  overlay-points  将路网点叠加到占据图上，生成 ${OVERLAY_IMG_WITH_POINTS_HUMAN}
  trans-json-coordinates
                  批量将 config.json 中列出的 points/routes 从 PNG 像素坐标转换为 Isaac Sim 坐标（用于旧数据迁移）

所有子命令都支持在后面追加原生 python 参数，例如：
  bash source/map_data_generation/map_tools.sh gen-routes --interp-step 3.0
  bash source/map_data_generation/map_tools.sh gen-routes --k-neighbors 12 --max-neighbor-dist 300
EOF
    ;;

  *)
    echo "未知子命令: ${cmd}" >&2
    echo "请使用: bash source/map_data_generation/map_tools.sh --help 查看帮助" >&2
    exit 1
    ;;
esac

