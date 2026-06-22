from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from rembg import remove
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import base64
import io
from pathlib import Path
from typing import Optional

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


@app.post("/generate-id")
async def generate_id(
    photo: UploadFile = File(...),
    student_no: str = Form(...),
    full_name: str = Form(...),
    course_year: str = Form(...),
    signature: Optional[str] = Form(None),
):
    # ─── 1. Read & remove background ─────────────────────────────────────────
    input_bytes = await photo.read()
    output_bytes = remove(input_bytes)                          # rembg → RGBA, bg transparent

    student = Image.open(io.BytesIO(output_bytes)).convert("RGBA")

    # Crop away excess transparent pixels so we only keep the person
    bbox = student.getbbox()
    if bbox:
        student = student.crop(bbox)

    # ─── 2. Open the ID template ──────────────────────────────────────────────
    template = Image.open(TEMPLATE_PATH).convert("RGBA")
    tw, th = template.size   # 644 × 1024

    # ─── 3. Photo placement ───────────────────────────────────────────────────
    # Header ("BUTUAN CITY COLLEGES INC.") ends at y=160 (black border line).
    # Keep photo strictly below y=175 so the header is never covered.
    PHOTO_TOP    = 175
    PHOTO_BOTTOM = 695
    ZONE_H       = PHOTO_BOTTOM - PHOTO_TOP   # 520 px

    max_photo_w = int(tw * 0.85)

    sw, sh = student.size
    scale  = min(max_photo_w / sw, ZONE_H / sh)
    new_w  = int(sw * scale)
    new_h  = int(sh * scale)
    student = student.resize((new_w, new_h), Image.LANCZOS)

    # Centre horizontally AND vertically inside the building zone
    x = (tw - new_w) // 2
    y = PHOTO_TOP + (ZONE_H - new_h) // 2

    template.paste(student, (x, y), student)

    # ─── 3b. Overlay digital signature ───────────────────────────────────────
    if signature:
        # Strip the data-URL prefix ("data:image/png;base64,…")
        raw = signature.split(",", 1)[1] if "," in signature else signature
        sig_img = Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGBA")

        # Make near-white pixels transparent so the signature floats cleanly
        sig_arr = np.array(sig_img)
        white = (sig_arr[:, :, 0] > 190) & (sig_arr[:, :, 1] > 190) & (sig_arr[:, :, 2] > 190)
        sig_arr[white, 3] = 0
        sig_img = Image.fromarray(sig_arr)

        # Scale to 48 % of card width, preserve aspect ratio
        sig_w = int(tw * 0.48)
        sig_h = int(sig_img.height * sig_w / sig_img.width)
        sig_img = sig_img.resize((sig_w, sig_h), Image.LANCZOS)

        # Place in the lower portion of the photo zone, horizontally centred
        signature_bottom_margin = 12
        sig_x = (tw - sig_w) // 2
        sig_y = PHOTO_BOTTOM - sig_h - signature_bottom_margin
        template.paste(sig_img, (sig_x, sig_y), sig_img)

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

    font_name   = load_font("arialbd.ttf", 36)   # name   – large bold
    font_studno = load_font("arialbd.ttf", 36)   # student no – bold
    font_course = load_font("arialbd.ttf", 34)   # course – large bold

    cx = tw // 2

    # ── Red band  y=695–768  centre y=731 ──────────────────────────────────────
    draw.text((cx, 731), full_name.upper(), font=font_name, fill="white", anchor="mm")

    # ── Yellow band  y=768–823  centre y=795 ───────────────────────────────────
    # Pre-printed "STUDENT NO.:" spans x=90–336 (past card centre).
    # Use numpy to directly overwrite every pixel in the band with the exact
    # yellow colour — this is guaranteed to erase the pre-printed text cleanly.
    arr = np.array(template)
    arr[768:824, :] = [255, 181, 13, 255]   # R,G,B,A — fully opaque yellow
    template = Image.fromarray(arr)
    draw = ImageDraw.Draw(template)          # re-bind draw to the updated image
    draw.text(
        (cx, 795),
        f"STUDENT NO.: {student_no}",
        font=font_studno,
        fill="black",
        anchor="mm"
    )

    # ── Dark-red section  y=911–1024  centre y=967 ─────────────────────────────
    draw.text((cx, 967), course_year.upper(), font=font_course, fill="white", anchor="mm")

    # ─── 5. Return the finished image ────────────────────────────────────────
    out_stream = io.BytesIO()
    template.save(out_stream, format="PNG")
    out_stream.seek(0)

    return StreamingResponse(out_stream, media_type="image/png")
