import os
import argparse
import time
import csv
import cv2
import numpy as np
from ultralytics import YOLO


# CLI
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--conf', type=float, default=0.5)
    parser.add_argument('--iou', type=float, default=0.2)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--save-img', action='store_true')
    return parser.parse_args()

# Utils
def compute_iou(box1, box2):
    box1 = list(map(float, box1))
    box2 = list(map(float, box2))

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0.0, (box1[2] - box1[0])) * max(0.0, (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0])) * max(0.0, (box2[3] - box2[1]))

    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0

def yolo_to_xyxy(box, w, h):
    x, y, bw, bh = box
    x1 = (x - bw/2) * w
    y1 = (y - bh/2) * h
    x2 = (x + bw/2) * w
    y2 = (y + bh/2) * h
    return [x1, y1, x2, y2]

def load_ground_truth(label_path, img_shape):
    h, w = img_shape[:2]
    gts = []

    if not os.path.exists(label_path):
        return gts

    with open(label_path, 'r') as f:
        for line in f.readlines():
            cls, x, y, bw, bh = map(float, line.strip().split())
            gts.append({
                "bbox": yolo_to_xyxy([x, y, bw, bh], w, h),
                "class": int(cls)
            })
    return gts

def match_predictions(preds, gts, iou_thresh):
    TP, FP = 0, 0
    matched = set()
    confs = []

    #preds = sorted(preds, key=lambda x: x["conf"], reverse=True)
    preds = sorted(preds, key=lambda x: x["conf"], reverse=True)[:10]

    for pred in preds:
        best_iou = 0
        best_gt = -1

        for i, gt in enumerate(gts):
            if i in matched:
                continue
            if pred["class"] != gt["class"]:
                continue

            iou = compute_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt = i

        if best_iou >= iou_thresh:
            TP += 1
            matched.add(best_gt)
            confs.append(pred["conf"])
        else:
            FP += 1

    FN = len(gts) - len(matched)
    return TP, FP, FN, confs

# Avaliar
def evaluate(model, data_path, args, output_dir):
    test_images = os.path.join(data_path, "test/images")
    test_labels = os.path.join(data_path, "test/labels")

    image_files = os.listdir(test_images)

    TP = FP = FN = 0
    all_confs = []

    start_time = time.time()

    for img_name in image_files:
        img_path = os.path.join(test_images, img_name)
        label_path = os.path.join(test_labels, img_name.replace('.jpg', '.txt'))

        img = cv2.imread(img_path)
        if img is None:
            continue

        results = model(img, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]

        preds = []
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                preds.append({
                    "bbox": [x1, y1, x2, y2],
                    "conf": conf,
                    "class": cls
                })

        gts = load_ground_truth(label_path, img.shape)

        tp, fp, fn, confs = match_predictions(preds, gts, args.iou)
        
        # DEBUG - primeira imagem
        if TP + FP + FN == 0:
            print("\nDEBUG PRIMEIRA IMAGEM")
            print("IMG SHAPE:", img.shape)
        
            if len(gts) > 0:
                print("GT[0]:", gts[0]["bbox"])
        
            if len(preds) > 0:
                print("PRED[0]:", preds[0]["bbox"])
                
            iou_test = compute_iou(preds[0]["bbox"], gts[0]["bbox"])
            print("IoU TESTE:", iou_test)
        
            print("\n")

        TP += tp
        FP += fp
        FN += fn
        all_confs.extend(confs)

        if args.save_img:
            draw_and_save(img, preds, gts, output_dir, img_name)

    total_time = time.time() - start_time
    fps = len(image_files) / total_time if total_time > 0 else 0

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    mean_conf = np.mean(all_confs) if all_confs else 0

    return precision, recall, f1, mean_conf, fps, TP, FP, FN

# Visual
def draw_and_save(img, preds, gts, output_dir, name):
    img_draw = img.copy()

    for gt in gts:
        x1, y1, x2, y2 = map(int, gt["bbox"])
        cv2.rectangle(img_draw, (x1,y1), (x2,y2), (255,0,0), 2)

    for pred in preds:
        x1, y1, x2, y2 = map(int, pred["bbox"])
        cv2.rectangle(img_draw, (x1,y1), (x2,y2), (0,0,255), 2)

    save_path = os.path.join(output_dir, "images")
    os.makedirs(save_path, exist_ok=True)
    cv2.imwrite(os.path.join(save_path, name), img_draw)

# Métricas
def save_metrics(output_dir, args, metrics):
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "metrics.csv")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["model", "precision", "recall", "f1", "mean_conf", "fps", "tp", "fp", "fn"])
        writer.writerow([args.name, *metrics])

# Main
def main():
    args = parse_args()

    model = YOLO(args.model)

    output_dir = os.path.join("outputs", args.name)

    metrics = evaluate(model, args.data, args, output_dir)

    save_metrics(output_dir, args, metrics)

    print("\nRESULTADOS:")
    print(f"Precision: {metrics[0]:.4f}")
    print(f"Recall:    {metrics[1]:.4f}")
    print(f"F1-score:  {metrics[2]:.4f}")
    print(f"FPS:       {metrics[4]:.2f}")

if __name__ == "__main__":
    import sys

    # Avaliação sem augmentation
    #sys.argv = ["evaluate.py", "--model", "runs/detect/runs/train/baseline_no_aug/weights/best.pt", "--data", "../data/drone_dataset", "--name", "no_augmentation", "--save-img"]

    # Avaliação com augmentation
    sys.argv = ["evaluate.py", "--model", "runs/detect/runs/train/baseline_with_aug/weights/best.pt", "--data", "../data/drone_dataset", "--name", "with_augmentation", "--save-img"]

    main()
