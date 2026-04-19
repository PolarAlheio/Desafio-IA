import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Drone Detection Training")

    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--data", type=str, default="../data/drone_dataset/data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--name", type=str, default="baseline")

    # Definir augmentation
    parser.add_argument("--augment", type=lambda x: x.lower() == "true", default=False,
                        help="Ativar ou desativar data augmentation")

    return parser.parse_args()


def main():
    args = parse_args()

    print("Iniciando treino YOLOv8")
    print(f"Modelo: {args.model}")
    print(f"Augmentation: {args.augment}")

    model = YOLO(args.model)

    # Configuração do augmentation
    if not args.augment:
        train_params = {
            "data": args.data,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "name": args.name,
            "project": "runs/train",
            "exist_ok": True,

            # Sem augmentation
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "degrees": 0.0,
            "translate": 0.0,
            "scale": 0.0,
            "shear": 0.0,
            "flipud": 0.0,
            "fliplr": 0.0,
            "mosaic": 0.0,
            "mixup": 0.0
        }
    else:
        # Default (com augmentation)
        train_params = {
            "data": args.data,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "name": args.name,
            "project": "runs/train",
            "exist_ok": True
        }

    model.train(**train_params)

    print("Treinamento finalizado")


if __name__ == "__main__":
    import sys
    #sys.argv = ["train.py", "--augment", "False", "--name", "baseline_no_aug"]
    sys.argv = ["train.py", "--augment", "True", "--name", "baseline_with_aug"]
    main()
    
