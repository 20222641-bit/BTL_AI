# Controller/ctl.py
from __future__ import annotations
from pathlib import Path
import re
import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO
import easyocr


# ================= Paths =================
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # .../BTL_Ai
RUNS_DIR     = PROJECT_ROOT / "runs"
MODEL_DIR    = PROJECT_ROOT / "Model" / "weights"


# ================= Plate rules =================
# Biểu thức vừa để chấm điểm, vừa để lọc chuỗi hợp lý
VN_PLATE_REGEX = re.compile(r'([1-9]\d)[A-Z]{1,2}\d{4,5}')


# ================= Utils =================
def _find_latest_best(runs_root: Path) -> Path | None:
    cands = list((runs_root / "detect").glob("*/weights/best.pt"))
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def _normalize_text(s: str) -> str:
    """Chuẩn hoá chuỗi OCR, bỏ khoảng trắng/ký tự ngăn cách, sửa các lỗi thường gặp."""
    s = s.upper().replace(" ", "").replace("-", "").replace(".", "")
    # thay thế các ký tự hay nhầm lẫn
    s = (s.replace("O", "0").replace("Q", "0")
           .replace("I", "1").replace("L", "1")
           .replace("Z", "2").replace("S", "5"))
    return s


def _score_text(s: str) -> float:
    s0 = _normalize_text(s)
    # độ dài càng gần format thật càng tốt + khớp regex được cộng điểm mạnh
    return (len(s0) / 10.0) + (1.0 if VN_PLATE_REGEX.search(s0) else 0.0)


def _crop_expand(img: np.ndarray, xyxy, expand: float = 0.08) -> np.ndarray:
    """Cắt bbox và nới biên (6–12% tùy dataset)."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, xyxy)
    dw, dh = int((x2 - x1) * expand), int((y2 - y1) * expand)
    x1 = max(0, x1 - dw); y1 = max(0, y1 - dh)
    x2 = min(w - 1, x2 + dw); y2 = min(h - 1, y2 + dh)
    return img[y1:y2, x1:x2]


def _prep_variants(crop_bgr: np.ndarray) -> list[np.ndarray]:
    """
    Tạo các biến thể tăng cường cho OCR:
      - Gray + CLAHE
      - Adaptive Threshold (đen/trắng)
      - Morphology close nhẹ
    """
    out: list[np.ndarray] = []
    h, w = crop_bgr.shape[:2]
    scale = max(2.0, 320 / max(1, min(h, w)))  # cạnh ngắn tối thiểu ~320 px
    big = cv2.resize(crop_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    g = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    out.append(g)

    thr = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 10)
    out.append(thr)
    out.append(255 - thr)

    k = np.ones((2, 2), np.uint8)
    out.append(cv2.morphologyEx(thr, cv2.MORPH_CLOSE, k, iterations=1))
    out.append(cv2.morphologyEx(255 - thr, cv2.MORPH_CLOSE, k, iterations=1))
    return out


def _rotate_variants(img_bgr: np.ndarray) -> list[np.ndarray]:
    """Trả về [gốc, +90°, -90°] để chống trường hợp ảnh dựng đứng."""
    return [
        img_bgr,
        cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]


def _ocr_easy_multi(reader: easyocr.Reader, images: list[np.ndarray]) -> str:
    """
    Thử OCR trên nhiều biến thể (xoay 0/±90 và tiền xử lý),
    chọn chuỗi có điểm cao nhất theo _score_text.
    """
    best_text, best_score = "", -1.0
    for im in images:
        for pre in _prep_variants(im):
            res = reader.readtext(
                pre, detail=1,
                allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                paragraph=False, text_threshold=0.5, low_text=0.3, link_threshold=0.3,
            )
            raw = " ".join([t[1] for t in res]) if res else ""
            sc = _score_text(raw)
            if sc > best_score:
                best_score, best_text = sc, raw
    return best_text


def _ocr_line_multi(reader: easyocr.Reader, img_bgr: np.ndarray, allowlist: str) -> str:
    """OCR một dòng với allowlist riêng (ví dụ trên: cho phép '-', dưới: cho phép '.')."""
    best_text, best_score = "", -1.0
    for pre in _prep_variants(img_bgr):
        res = reader.readtext(
            pre, detail=1,
            allowlist=allowlist,
            paragraph=False, text_threshold=0.5, low_text=0.3, link_threshold=0.3,
        )
        raw = " ".join([t[1] for t in res]) if res else ""
        sc = _score_text(raw)
        if sc > best_score:
            best_score, best_text = sc, raw
    return best_text


def _split_two_lines(crop_bgr: np.ndarray):
    """Ước lượng vị trí cắt giữa 2 dòng nếu là biển 2 tầng."""
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    binv = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )
    hist = binv.sum(axis=1).astype(np.float32)
    h = binv.shape[0]
    s, e = int(0.35 * h), int(0.65 * h)
    if e - s < 10:
        return None, None
    cut = s + int(np.argmin(hist[s:e]))
    if cut <= 10 or h - cut <= 10:
        return None, None
    return crop_bgr[:cut, :], crop_bgr[cut:, :]


def _build_plate_from_lines(top_raw: str, bot_raw: str) -> str:
    """
    Ráp lại theo dạng VN: 2 số + 1-2 chữ + 1 số + 5 số.
    - Dòng trên cho phép '-', khi ráp sẽ bỏ '-'
    - Dòng dưới cho phép '.', khi ráp sẽ bỏ '.'
    """
    top = _normalize_text(top_raw)   # đã bỏ '-' rồi
    bot = _normalize_text(bot_raw)   # đã bỏ '.' rồi

    # Kì vọng top: 2 số đầu
    if len(top) < 4 or not top[:2].isdigit():
        # fallback thô
        return (top + bot)[:10]

    prov = top[:2]
    rest = top[2:]

    # bóc series (1-2 chữ) + 1 chữ số sau series
    letters = "".join([c for c in rest if c.isalpha() and c not in "FOIQ"])  # loại FOIQ dễ nhầm
    digits1 = "".join([c for c in rest if c.isdigit()])

    ser = letters[:2] if len(letters) >= 2 else (letters[:1] if len(letters) == 1 else "")
    d_after = digits1[0] if len(digits1) >= 1 else ""

    # dòng dưới: lấy 5 số cuối
    nums = "".join([c for c in bot if c.isdigit()])
    if len(nums) >= 5:
        last5 = nums[-5:]
    else:
        last5 = (digits1 + nums)[-5:].rjust(5, "0")  # bù tạm nếu thiếu

    plate = f"{prov}{ser}{d_after}{last5}"
    return plate


def _cv2_to_tk(img_bgr: np.ndarray, target_size=(480, 350)) -> ImageTk.PhotoImage:
    """Resize ảnh OpenCV -> PhotoImage (Tkinter)."""
    h, w = img_bgr.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h, 1.0)
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(img_rgb))


# ================= Controller =================
class Controller:
    """Kết nối GUI với YOLO + EasyOCR. Trả (crop_tk, final_text, vis_tk, conf)."""

    def __init__(self, model_path: str | None = None):
        # 1) Chọn weights
        if model_path:
            mp = Path(model_path)
        else:
            mp = _find_latest_best(RUNS_DIR)
            if not mp:
                cand = MODEL_DIR / "best.pt"
                mp = cand if cand.exists() else None

        if not mp or not mp.exists():
            raise FileNotFoundError(
                "Không tìm thấy best.pt.\n"
                f"- Đã tìm: {RUNS_DIR}/detect/*/weights/best.pt\n"
                f"- Và:     {MODEL_DIR}/best.pt\n"
                "Hãy truyền model_path cụ thể trong main.py hoặc kiểm tra lại đường dẫn."
            )

        self.model_path = mp
        self.model = YOLO(self.model_path.as_posix())

        # 2) EasyOCR: ưu tiên GPU, lỗi thì dùng CPU
        try:
            self.reader = easyocr.Reader(['en'], gpu=True)
        except Exception:
            self.reader = easyocr.Reader(['en'], gpu=False)

    def home(self):
        pass

    def detect_plate(self, file_path: str):
        """Detect -> crop (nới nhẹ) -> OCR ưu tiên 2 dòng -> trả (crop_tk, text, vis_tk, conf)."""
        img = cv2.imread(file_path)
        if img is None:
            raise FileNotFoundError("Không đọc được ảnh.")

        # 1) Detect
        det = self.model.predict(source=img, conf=0.25, device=0, verbose=False)[0]
        if len(det.boxes) == 0:
            return None, "", None, 0.0

        i_best = int(det.boxes.conf.argmax())
        xyxy = det.boxes.xyxy[i_best].tolist()
        conf = float(det.boxes.conf[i_best])

        # 2) Crop (KHÔNG deskew)
        crop = _crop_expand(img, xyxy, expand=0.08)

        # 3) Ưu tiên tách 2 dòng
        top_img, bot_img = _split_two_lines(crop)
        if top_img is not None and bot_img is not None:
            # Dòng trên: cho phép '-', để OCR không cố biến '-' thành chữ
            top_text = _ocr_line_multi(self.reader, top_img,
                                       allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-')
            # Dòng dưới: chỉ số + dấu chấm
            bot_text = _ocr_line_multi(self.reader, bot_img,
                                       allowlist='0123456789.')
            final_text = _build_plate_from_lines(top_text, bot_text)

        else:
            # Fallback 1 khối (thử 0/±90)
            candidates = _rotate_variants(crop)
            text_raw = _ocr_easy_multi(self.reader, candidates)

            # Thử tách 2 dòng lại nếu OCR khối còn yếu
            if len(_normalize_text(text_raw)) < 6:
                t, b = _split_two_lines(crop)
                if t is not None:
                    t1 = _ocr_line_multi(self.reader, t,  allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-')
                    t2 = _ocr_line_multi(self.reader, b,  allowlist='0123456789.')
                    text_raw = _build_plate_from_lines(t1, t2)

            final_text = _normalize_text(text_raw)

        # 4) Vẽ vis cho GUI
        x1, y1, x2, y2 = map(int, xyxy)
        vis = img.copy()
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, final_text, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        crop_tk = _cv2_to_tk(crop, target_size=(480, 350))
        vis_tk  = _cv2_to_tk(vis,  target_size=(480, 350))
        return crop_tk, final_text, vis_tk, conf
