import cv2
import pytesseract
import subprocess
from datetime import datetime
from pathlib import Path

CUSTOM_OCR_CONFIG = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/'

BASE_DIR = Path(__file__).resolve().parent
CAPTURE_DIR = (BASE_DIR / "data_capture")
CAPTURE_DIR.mkdir(exist_ok=True)

def capture_with_fswebcam():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = CAPTURE_DIR / f"full_{ts}.jpg"

    cmd = [
        "fswebcam",
        "--no-banner",
        "-r", "1280x720",
        str(img_path)
    ]

    print("📸 Acquisizione immagine...")
    subprocess.run(cmd, check=True)

    frame = cv2.imread(str(img_path))
    if frame is None:
        raise RuntimeError("Errore nel caricamento dell'immagine.")
    return frame, img_path

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def main():
    frame, frame_path = capture_with_fswebcam()
    proc = preprocess(frame)

    h, w = proc.shape
    roi = proc[int(h * 0.1):int(h * 0.4), int(w * 0.05):int(w * 0.6)]

    text = pytesseract.image_to_string(roi, config=CUSTOM_OCR_CONFIG).strip()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    roi_path = CAPTURE_DIR / f"roi_{ts}.jpg"
    cv2.imwrite(str(roi_path), roi)

    print("------ RISULTATI ------")
    print(f"🔤 Testo rilevato (Tesseract): {text}")
    print(f"📁 Immagine completa salvata: {frame_path.name}")
    print(f"📁 ROI salvata in:            {roi_path.name}")
    print("------------------------")

if __name__ == "__main__":
    main()
