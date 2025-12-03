import torch
import cv2
from pathlib import Path
from train_crnn import CRNN, IDX_TO_CHAR, ALPHABET

MODEL_PATH = Path("../app/models/crnn_best.pt")

def decode(logits):
    preds = logits.argmax(2).transpose(1,0).squeeze()
    out = []
    last = None
    for p in preds:
        if p != last and p > 0:
            out.append(IDX_TO_CHAR[p.item()])
        last = p
    return "".join(out)

def main():
    model = CRNN(num_classes=len(ALPHABET)+1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    cap = cv2.VideoCapture(0)

    while True:
        ret, f = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(gray, (200, 50))
        tens = torch.tensor(img/255.0).unsqueeze(0).unsqueeze(0).float()

        with torch.no_grad():
            logits = model(tens)
            text = decode(logits)

        cv2.putText(f, text, (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

        cv2.imshow("OCR LIVE", f)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
