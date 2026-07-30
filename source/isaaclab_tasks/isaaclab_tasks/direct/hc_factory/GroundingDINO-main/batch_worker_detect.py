import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.ops import nms

from groundingdino.util.inference import load_model, load_image, predict


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# 工人编号与衣服颜色映射：
# 0 红色
# 1 绿色
# 2 浅蓝色
# 3 黄色
# 4 深蓝色
WORKER_COLOR_MAP = {
    "red": "worker_0",
    "green": "worker_1",
    "light_blue": "worker_2",
    "yellow": "worker_3",
    "dark_blue": "worker_4",
    "unknown": "unknown",
}


def list_images(input_dir: str):
    input_path = Path(input_dir)
    image_paths = []
    for p in input_path.rglob("*"):
        if p.suffix.lower() in IMAGE_EXTS:
            image_paths.append(p)
    return sorted(image_paths)


def cxcywh_to_xyxy(boxes, width, height):
    """
    GroundingDINO 输出通常是归一化 cxcywh。
    转为像素坐标 xyxy。
    """
    boxes = boxes.clone()
    boxes_xyxy = torch.zeros_like(boxes)

    boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2.0) * width
    boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2.0) * height
    boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2.0) * width
    boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2.0) * height

    boxes_xyxy[:, 0::2] = boxes_xyxy[:, 0::2].clamp(0, width - 1)
    boxes_xyxy[:, 1::2] = boxes_xyxy[:, 1::2].clamp(0, height - 1)

    return boxes_xyxy


def crop_upper_body(image_rgb: np.ndarray, box_xyxy):
    """
    从人体检测框中裁剪上半身区域。
    不使用整个 bbox，是为了减少裤子、鞋、地面、背景干扰。
    """
    h_img, w_img = image_rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]

    x1 = max(0, min(x1, w_img - 1))
    x2 = max(0, min(x2, w_img - 1))
    y1 = max(0, min(y1, h_img - 1))
    y2 = max(0, min(y2, h_img - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    w = x2 - x1
    h = y2 - y1

    # 取人体框中上半身区域
    tx1 = int(x1 + 0.20 * w)
    tx2 = int(x1 + 0.80 * w)
    ty1 = int(y1 + 0.20 * h)
    ty2 = int(y1 + 0.65 * h)

    torso = image_rgb[ty1:ty2, tx1:tx2]

    if torso.size == 0:
        return None

    return torso


def classify_color_hsv(torso_rgb: np.ndarray):
    """
    基于 HSV 主色判断衣服颜色。

    输出：
    red        -> worker_0
    green      -> worker_1
    light_blue -> worker_2
    yellow     -> worker_3
    dark_blue  -> worker_4
    """
    if torso_rgb is None or torso_rgb.size == 0:
        return "unknown", 0.0

    hsv = cv2.cvtColor(torso_rgb, cv2.COLOR_RGB2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # 过滤低饱和度、过暗、过亮像素
    valid = (s > 45) & (v > 45) & (v < 250)

    if valid.sum() < 20:
        return "unknown", 0.0

    h_valid = h[valid]
    s_valid = s[valid]
    v_valid = v[valid]
    total = len(h_valid)

    # OpenCV HSV 中 H 范围为 0~179
    red_mask = (h_valid <= 10) | (h_valid >= 170)
    yellow_mask = (h_valid > 20) & (h_valid <= 38)
    green_mask = (h_valid > 38) & (h_valid <= 85)

    # 浅蓝：亮度更高，Hue 多在青蓝范围
    light_blue_mask = (
        ((h_valid > 85) & (h_valid <= 115) & (v_valid >= 135))
        |
        ((h_valid > 85) & (h_valid <= 125) & (v_valid >= 150) & (s_valid <= 210))
    )

    # 深蓝：亮度更低，Hue 更偏蓝或蓝紫
    dark_blue_mask = (
        ((h_valid > 105) & (h_valid <= 135) & (v_valid < 170))
        |
        ((h_valid > 115) & (h_valid <= 145))
    )

    color_masks = {
        "red": red_mask,
        "green": green_mask,
        "light_blue": light_blue_mask,
        "yellow": yellow_mask,
        "dark_blue": dark_blue_mask,
    }

    ratios = {
        color: float(mask.sum()) / float(total)
        for color, mask in color_masks.items()
    }

    best_color = max(ratios, key=ratios.get)
    confidence = ratios[best_color]

    # 如果浅蓝/深蓝不稳定，用蓝色像素平均亮度二次判断
    blue_mask = (h_valid > 85) & (h_valid <= 145)
    blue_ratio = float(blue_mask.sum()) / float(total)

    if best_color in ["light_blue", "dark_blue"] and confidence < 0.22 and blue_ratio > 0.20:
        mean_v_blue = float(v_valid[blue_mask].mean())
        if mean_v_blue >= 145:
            return "light_blue", blue_ratio
        else:
            return "dark_blue", blue_ratio

    if confidence < 0.18:
        return "unknown", confidence

    return best_color, confidence


def draw_annotations(image_rgb, detections):
    """
    在图片上画检测框、颜色和工人编号。
    """
    pil_img = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_img)

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        label = (
            f'{det["worker_id"]} | {det["color"]} | '
            f'det={det["det_score"]:.2f} | color={det["color_confidence"]:.2f}'
        )

        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)

        text_bbox = draw.textbbox((x1, y1), label)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]

        y_text = max(0, y1 - th - 4)
        draw.rectangle([x1, y_text, x1 + tw + 4, y1], fill=(255, 0, 0))
        draw.text((x1 + 2, y_text + 1), label, fill=(255, 255, 255))

    return np.array(pil_img)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"[INFO] device = {device}")

    print("[INFO] loading GroundingDINO model...")
    model = load_model(args.config, args.weights)
    model = model.to(device)

    image_paths = list_images(args.input_dir)
    print(f"[INFO] found {len(image_paths)} images under {args.input_dir}")

    output_dir = Path(args.output_dir)
    annotated_dir = output_dir / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "results.csv"
    json_path = output_dir / "results.json"

    all_results = []

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow([
            "image_path",
            "relative_path",
            "has_worker",
            "num_workers",
            "worker_index",
            "worker_id",
            "color",
            "det_score",
            "color_confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "phrase",
        ])

        for idx, image_path in enumerate(image_paths):
            print(f"[{idx + 1}/{len(image_paths)}] {image_path}")

            image_source, image = load_image(str(image_path))
            h, w = image_source.shape[:2]

            boxes, logits, phrases = predict(
                model=model,
                image=image,
                caption=args.prompt,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                device=device,
            )

            detections = []

            if boxes is not None and len(boxes) > 0:
                boxes_xyxy = cxcywh_to_xyxy(boxes.cpu(), w, h)
                scores = logits.cpu()

                # NMS 去掉重复框
                keep = nms(boxes_xyxy, scores, args.nms_threshold)
                boxes_xyxy = boxes_xyxy[keep]
                scores = scores[keep]
                kept_phrases = [phrases[i] for i in keep.tolist()]

                for j, (box, score, phrase) in enumerate(zip(boxes_xyxy, scores, kept_phrases)):
                    box_list = [float(v) for v in box.tolist()]

                    torso = crop_upper_body(image_source, box_list)
                    color, color_conf = classify_color_hsv(torso)
                    worker_id = WORKER_COLOR_MAP.get(color, "unknown")

                    detections.append({
                        "worker_index": j,
                        "bbox": box_list,
                        "det_score": float(score.item()),
                        "phrase": phrase,
                        "color": color,
                        "color_confidence": float(color_conf),
                        "worker_id": worker_id,
                    })

            has_worker = len(detections) > 0

            try:
                rel_path = image_path.relative_to(Path(args.input_dir))
            except ValueError:
                rel_path = image_path.name

            annotated_path = annotated_dir / rel_path
            annotated_path.parent.mkdir(parents=True, exist_ok=True)

            if args.save_annotated:
                annotated_img = draw_annotations(image_source, detections)
                cv2.imwrite(
                    str(annotated_path),
                    cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR),
                )

            image_result = {
                "image_path": str(image_path),
                "relative_path": str(rel_path),
                "has_worker": has_worker,
                "num_workers": len(detections),
                "detections": detections,
                "annotated_path": str(annotated_path) if args.save_annotated else None,
            }
            all_results.append(image_result)

            if not detections:
                writer.writerow([
                    str(image_path),
                    str(rel_path),
                    False,
                    0,
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ])
            else:
                for det in detections:
                    x1, y1, x2, y2 = det["bbox"]
                    writer.writerow([
                        str(image_path),
                        str(rel_path),
                        True,
                        len(detections),
                        det["worker_index"],
                        det["worker_id"],
                        det["color"],
                        det["det_score"],
                        det["color_confidence"],
                        x1,
                        y1,
                        x2,
                        y2,
                        det["phrase"],
                    ])

    save_json(json_path, all_results)

    print("[DONE]")
    print(f"[INFO] CSV saved to: {csv_path}")
    print(f"[INFO] JSON saved to: {json_path}")

    if args.save_annotated:
        print(f"[INFO] annotated images saved to: {annotated_dir}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs_worker_detection")

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)

    parser.add_argument("--prompt", type=str, default="person . worker . human .")
    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--nms_threshold", type=float, default=0.50)

    parser.add_argument("--save_annotated", action="store_true")
    parser.add_argument("--cpu", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())