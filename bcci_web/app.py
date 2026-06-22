from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from rembg import remove
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import base64
import io
from pathlib import Path
from typing import Optional


#API start command (run in terminal):
#cd C:\xampp\htdocs\try\bcci_web
#python -m uvicorn app:app --host 127.0.0.1 --port 8010

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATE_DIR = BASE_DIR / "templates"
TEMPLATE_PATH = STATIC_DIR / "BCCI ID 2026-2027 FRONT3.png"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def home():
    return FileResponse(TEMPLATE_DIR / "test.html")


@app.get("/test")
def test_page():
    return FileResponse(TEMPLATE_DIR / "test.html")


@app.post("/remove-bg")
async def remove_bg_endpoint(photo: UploadFile = File(...)):
    input_bytes = await photo.read()
    output_bytes = remove(input_bytes)
    student = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
    bbox = student.getbbox()
    if bbox:
        student = student.crop(bbox)
    buf = io.BytesIO()
    student.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return JSONResponse({
        "image": f"data:image/png;base64,{b64}",
        "width": student.width,
        "height": student.height,
    })


@app.post("/generate-id")
async def generate_id(
    photo: Optional[UploadFile] = File(None),
    photo_data: Optional[str] = Form(None),
    photo_x: Optional[int] = Form(None),
    photo_y: Optional[int] = Form(None),
    photo_w: Optional[int] = Form(None),
    photo_h: Optional[int] = Form(None),
    sig_x: Optional[int] = Form(None),
    sig_y: Optional[int] = Form(None),
    sig_w: Optional[int] = Form(None),
    sig_h: Optional[int] = Form(None),
    student_no: str = Form(...),
    full_name: str = Form(...),
    course_year: str = Form(...),
    signature: Optional[str] = Form(None),
):
    # ─── 1. Get student photo ─────────────────────────────────────────────────
    if photo_data:
        raw = photo_data.split(",", 1)[1] if "," in photo_data else photo_data
        student = Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGBA")
    elif photo and photo.filename:
        input_bytes = await photo.read()
        output_bytes = remove(input_bytes)
        student = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
        bbox = student.getbbox()
        if bbox:
            student = student.crop(bbox)
    else:
        raise HTTPException(status_code=400, detail="No photo provided")

    # ─── 2. Open the ID template ──────────────────────────────────────────────
    template = Image.open(TEMPLATE_PATH).convert("RGBA")
    tw, _ = template.size   # 644 × 1024

    # ─── 3. Photo placement ───────────────────────────────────────────────────
    PHOTO_TOP    = 175
    PHOTO_BOTTOM = 695
    ZONE_H       = PHOTO_BOTTOM - PHOTO_TOP   # 520 px

    if photo_w is not None and photo_h is not None:
        new_w, new_h = photo_w, photo_h
        student = student.resize((new_w, new_h), Image.LANCZOS)
        x = photo_x if photo_x is not None else (tw - new_w) // 2
        y = photo_y if photo_y is not None else PHOTO_TOP + (ZONE_H - new_h) // 2
    else:
        max_photo_w = int(tw * 0.85)
        sw, sh = student.size
        scale  = min(max_photo_w / sw, ZONE_H / sh)
        new_w  = int(sw * scale)
        new_h  = int(sh * scale)
        student = student.resize((new_w, new_h), Image.LANCZOS)
        x = (tw - new_w) // 2
        y = PHOTO_TOP + (ZONE_H - new_h) // 2

    template.paste(student, (x, y), student)

    # ─── 3b. Overlay digital signature ───────────────────────────────────────
    if signature:
        raw = signature.split(",", 1)[1] if "," in signature else signature
        sig_img = Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGBA")

        sig_arr = np.array(sig_img)
        white = (sig_arr[:, :, 0] > 190) & (sig_arr[:, :, 1] > 190) & (sig_arr[:, :, 2] > 190)
        sig_arr[white, 3] = 0
        sig_img = Image.fromarray(sig_arr)

        if sig_w is not None and sig_h is not None:
            sig_img = sig_img.resize((sig_w, sig_h), Image.LANCZOS)
            sx = sig_x if sig_x is not None else (tw - sig_w) // 2
            sy = sig_y if sig_y is not None else PHOTO_BOTTOM - sig_h - 12
        else:
            default_sig_w = int(tw * 0.48)
            default_sig_h = int(sig_img.height * default_sig_w / sig_img.width)
            sig_img = sig_img.resize((default_sig_w, default_sig_h), Image.LANCZOS)
            sx = (tw - default_sig_w) // 2
            sy = PHOTO_BOTTOM - default_sig_h - 12

        template.paste(sig_img, (sx, sy), sig_img)

    # ─── 4. Draw text ────────────────────────────────────────────────────────
    draw = ImageDraw.Draw(template)

    def load_font(name, size):
        for attempt in [name, "arial.ttf", None]:
            try:
                if attempt:
                    return ImageFont.truetype(attempt, size)
            except Exception:
                pass
        return ImageFont.load_default()

    font_name   = load_font("arialbd.ttf", 36)
    font_studno = load_font("arialbd.ttf", 36)
    font_course = load_font("arialbd.ttf", 34)

    cx = tw // 2

    draw.text((cx, 731), full_name.upper(), font=font_name, fill="white", anchor="mm")

    arr = np.array(template)
    arr[768:824, :] = [255, 181, 13, 255]
    template = Image.fromarray(arr)
    draw = ImageDraw.Draw(template)
    draw.text(
        (cx, 795),
        f"STUDENT NO.: {student_no}",
        font=font_studno,
        fill="black",
        anchor="mm"
    )

    draw.text((cx, 967), course_year.upper(), font=font_course, fill="white", anchor="mm")

    # ─── 5. Return the finished image ────────────────────────────────────────
    out_stream = io.BytesIO()
    template.save(out_stream, format="PNG")
    out_stream.seek(0)

    return StreamingResponse(out_stream, media_type="image/png")
