# predict_signs.py

import argparse
import os

import cv2
import torch
from ultralytics import YOLO

def main():
    # ── 1) Parsear args ──────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Detección de letras con YOLOv11 (imagen fija o cámara)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-i", "--image",  action="store_true",
                        help="Procesar la imagen fija definida en el código")
    group.add_argument("-v", "--video",  action="store_true",
                        help="Abrir la cámara web y procesar el stream en vivo")
    args = parser.parse_args()

    # ── 2) Configuración de dispositivo y modelo ────────────────
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    model = YOLO("runs/train/sign_detect_small/weights/best.pt")
    class_names = model.names  # índice→etiqueta (letra)

    IMG_SIZE = 640
    CONF_THRES = 0.25
    SAVE_DIR = "runs/detect/sign_preds"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── 3) Modo “imagen fija” ──────────────────────────────────
    if args.image:
        src = "data/test/images/A22_jpg.rf.f02ad8558ce1c88213b4f83c0bc66bc8.jpg"
        results = model.predict(
            source=src,
            device=device,
            imgsz=IMG_SIZE,
            conf=CONF_THRES,
            save=True,
            save_txt=False,
            save_conf=True,
            project="runs/detect",
            name="small",
            exist_ok=True,
        )

        print(f"\n-- Resultados para {os.path.basename(src)} --")
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                name = class_names[cls]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                print(f"{name} ({conf:.2f}) @ [{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]")

        print(f"\nImagen procesada ✅  Revisa: runs/detect/sign_preds/{os.path.basename(src)}")

    # ── 4) Modo “cámara web” ────────────────────────────────────
    elif args.video:
        print("Abrindo cámara web. Pulsa 'q' para salir.")
        stream = model.predict(
            source=0,
            device=device,
            imgsz=IMG_SIZE,
            conf=CONF_THRES,
            stream=True
        )

        # Crear ventana OpenCV
        cv2.namedWindow("YOLOv11 Cam", cv2.WINDOW_NORMAL)
        for r in stream:
            # r.plot() devuelve la imagen con las cajas dibujadas
            frame = r.plot()
            cv2.imshow("YOLOv11 Cam", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()
        print("Stream detenido. 👍")

if __name__ == "__main__":
    main()
