"""Local QR image decoding; does not access decoded links."""
from pathlib import Path

def decode_qr(path: str) -> dict:
    try:
        import cv2
        import numpy as np
    except ImportError as e:
        return {"ok": False, "error": "QR decoding requires opencv-python and numpy: " + str(e), "codes": []}
    try:
        data = Path(path).expanduser().read_bytes()
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, ValueError) as e:
        return {"ok": False, "error": str(e), "codes": []}
    if image is None:
        return {"ok": False, "error": "Unreadable image", "codes": []}
    detector = cv2.QRCodeDetector()
    codes = []
    try:
        found, texts, points, _ = detector.detectAndDecodeMulti(image)
        if found and points is not None:
            codes = [{"text": text, "points": poly.tolist()} for text, poly in zip(texts, points) if text]
    except cv2.error:
        pass
    if not codes:
        text, points, _ = detector.detectAndDecode(image)
        if text:
            codes = [{"text": text, "points": points.reshape(-1, 2).tolist() if points is not None else []}]
    return {"ok": True, "codes": codes, "message": "decoded" if codes else "No decodable QR code found"}
