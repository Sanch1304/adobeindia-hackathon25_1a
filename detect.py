# === PHASE 1: PDF TO BINARIZED IMAGES ===
print("\n📄 Converting PDF pages to B/W images...")
from pathlib import Path
import cv2
import numpy as np
import fitz  # PyMuPDF
import os
import time
import json
total_start = time.time()


PDF_PATH = "input/IBM_SPSS_Complex_Samples.pdf"
IMAGE_DIR = "../data/images"
OUTPUT_DIR = "../app/processing_data"
out = "/output"

DPI = 100

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(out, exist_ok=True)

start_pdf = time.time()
doc = fitz.open(PDF_PATH)
for i, page in enumerate(doc):
    mat = fitz.Matrix(DPI / 72, DPI / 72)
    pix = page.get_pixmap(matrix=mat)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    out_path = os.path.join(IMAGE_DIR, f"page_{i + 1}.png")
    cv2.imwrite(out_path, bw)
print(f"✅ PDF to images done in {time.time() - start_pdf:.2f}s")

# === PHASE 2: YOLO DETECTION + FILTERING + JSON ===
print("\n🤖 Running YOLOv10n inference and filtering for headings/titles...")
import onnxruntime
import json

ONNX_PATH = "models/yolov10n_best.onnx"
CONF_THRESH = 0.25
IOU_THRESH = 0.50
IMG_SIZE = (640, 480)
BATCH_SIZE = 50
ID2LABEL = {
    0: 'Caption', 1: 'Footnote', 2: 'Formula', 3: 'List-item', 4: 'Page-footer',
    5: 'Page-header', 6: 'Picture', 7: 'heading', 8: 'Table', 9: 'Text', 10: 'Title'
}
TARGET_CLASSES = [7, 10]  # only heading and title
JSON_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "rapprort.json")

session = onnxruntime.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

def preprocess(img_path):
    img = cv2.imread(str(img_path))
    orig = img.copy()
    img_resized = cv2.resize(img, IMG_SIZE)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_transposed = np.transpose(img_norm, (2, 0, 1))
    return img_transposed[np.newaxis, ...], orig, img.shape[1], img.shape[0]

def nms(boxes, scores, iou_threshold):
    if not boxes:
        return []
    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESH, iou_threshold)
    if isinstance(indices, np.ndarray):
        return indices.flatten().tolist()
    elif isinstance(indices, list):
        return [i[0] if isinstance(i, (list, tuple)) else i for i in indices]
    return []

def sort_key(path):
    stem = path.stem
    num = ''.join(filter(str.isdigit, stem))
    return int(num) if num.isdigit() else 0

image_paths = sorted(Path(IMAGE_DIR).glob("*.png"), key=sort_key)
ocr_results = {}

for i in range(0, len(image_paths), BATCH_SIZE):
    batch_paths = image_paths[i:i + BATCH_SIZE]
    batch_input, originals = [], []

    for path in batch_paths:
        tensor, orig_img, orig_w, orig_h = preprocess(path)
        batch_input.append(tensor)
        originals.append((path, orig_img, orig_w, orig_h))

    batch_input = np.concatenate(batch_input, axis=0)
    outputs = session.run(None, {input_name: batch_input})[0]

    for idx, (path, orig_img, orig_w, orig_h) in enumerate(originals):
        pred = outputs[idx]
        scale_w = orig_w / IMG_SIZE[0]
        scale_h = orig_h / IMG_SIZE[1]
        boxes, scores, classes = [], [], []

        for det in pred:
            x1, y1, x2, y2, conf, cls_id = det[:6]
            if conf < CONF_THRESH:
                continue
            cls_id = int(cls_id)
            x1 = int(x1 * scale_w)
            y1 = int(y1 * scale_h)
            x2 = int(x2 * scale_w)
            y2 = int(y2 * scale_h)
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(float(conf))
            classes.append(cls_id)

        selected = nms(boxes, scores, IOU_THRESH)
        page_number = int(path.stem.split("_")[-1])
        ocr_results.setdefault(page_number, [])

        for j in selected:
            x, y, w, h = boxes[j]
            x2 = x + w
            y2 = y + h
            cls_id = classes[j]
            label = ID2LABEL[cls_id]

            if cls_id == 7 and y2 > orig_h * 0.90:
                continue  # skip footer headings

            if cls_id in [7, 10]:  # heading or title
                too_close_to_picture = False
                for other_idx in selected:
                    if other_idx == j:
                        continue
                    ox, oy, ow, oh = boxes[other_idx]
                    ox2 = ox + ow
                    oy2 = oy + oh
                    o_cls = classes[other_idx]
                    if o_cls in [6, 8]:  # picture or table
                        vertical_gap = y - (oy + oh)
                        if 0 <= vertical_gap < 20 and ox < x2 and ox2 > x:
                            too_close_to_picture = True
                            break
                if too_close_to_picture:
                    continue
            
            if cls_id in TARGET_CLASSES:
                ocr_results[page_number].append({
                    "type": label,
                    "box": [x, y, x2, y2]
                })
            



            # # === NEW: Save YOLO predictions with all-class boxes ===
            # vis_img = orig_img.copy()
            # for k in range(len(boxes)):
            #     x, y, w, h = boxes[k]
            #     cls_id = classes[k]
            #     label = ID2LABEL.get(cls_id, str(cls_id))
            #     color = (0, 255, 0)  # default green

            #     # Optional: use class-specific colors
            #     if cls_id == 10:      # Title
            #         color = (0, 0, 255)  # Red
            #     elif cls_id == 7:      # Heading
            #         color = (255, 0, 0)  # Blue
            #     elif cls_id == 6:      # Picture
            #         color = (0, 255, 255)  # Yellow
            #     elif cls_id == 8:      # Table
            #         color = (255, 255, 0)  # Cyan

            #     cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, 1)
            #     cv2.putText(vis_img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # save_path = os.path.join(OUTPUT_DIR, "yolo_preds", f"{path.stem}_pred.png")
            # os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # cv2.imwrite(save_path, vis_img)


with open(JSON_OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(ocr_results, f, indent=2, ensure_ascii=False)

print(f"✅ YOLO detection & filtering complete. Results saved to: {JSON_OUTPUT_PATH}")

# === PHASE 3: PDF ANNOTATION OF DETECTIONS ===
print("\n✏️ Annotating original PDF with detected heading/title boxes...")
start_annot = time.time()
doc = fitz.open(PDF_PATH)
for page_num, items in ocr_results.items():
    if page_num - 1 >= len(doc):
        continue
    page = doc[page_num - 1]
    pix = page.get_pixmap(matrix=fitz.Matrix(DPI / 72, DPI / 72))
    img_width, img_height = pix.width, pix.height
    pdf_width, pdf_height = page.rect.width, page.rect.height

    for item in items:
        label = item.get("type", "").lower()
        if label not in ["heading", "title"]:
            continue
        x1, y1, x2, y2 = item["box"]
        rect = fitz.Rect(
            x1 * pdf_width / img_width,
            y1 * pdf_height / img_height,
            x2 * pdf_width / img_width,
            y2 * pdf_height / img_height,
        )
        page.draw_rect(rect, color=(0, 1, 0), width=1)
        page.insert_textbox(rect, label.upper(), fontsize=6, color=(1, 0, 0))

annotated_path = os.path.join(OUTPUT_DIR, "annotated.pdf")
doc.save(annotated_path)
doc.close()
print(f"✅ Annotated PDF saved at: {annotated_path}")
print(f"✅ PDF annotation done in {time.time() - start_annot:.2f}s")

# === PHASE 4: TEXT EXTRACTION WITH HEADING LEVEL CLASSIFICATION ===
print("\n🔍 Extracting training-format text and font info with normalization and level classification...")
import json

PADDING = 0.10  # 5% padding on each side
training_output_path = os.path.join(OUTPUT_DIR, "deeponet.json")
training_results = []
doc = fitz.open(PDF_PATH)

# === PASS 1: COLLECT ALL HEADING FONT SIZES ===
all_heading_sizes = []
for page_num, items in ocr_results.items():
    if page_num - 1 >= len(doc): continue
    page = doc[page_num - 1]
    pix = page.get_pixmap(matrix=fitz.Matrix(DPI / 72, DPI / 72))
    img_width, img_height = pix.width, pix.height
    pdf_width, pdf_height = page.rect.width, page.rect.height

    for item in items:
        if item["type"].lower() != "heading":
            continue
        x1, y1, x2, y2 = item["box"]
        w_pad = int((x2 - x1) * PADDING)
        h_pad = int((y2 - y1) * PADDING)
        x1_pad = max(x1 - w_pad, 0)
        y1_pad = max(y1 - h_pad, 0)
        x2_pad = min(x2 + w_pad, img_width)
        y2_pad = min(y2 + h_pad, img_height)

        rect = fitz.Rect(
            x1_pad * pdf_width / img_width,
            y1_pad * pdf_height / img_height,
            x2_pad * pdf_width / img_width,
            y2_pad * pdf_height / img_height,
        )

        spans = page.get_text("dict", clip=rect).get("blocks", [])
        for block in spans:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size = round(span.get("size", 0), 2)
                    all_heading_sizes.append(size)

max_heading_font_size = max(all_heading_sizes) if all_heading_sizes else 1.0

# === PASS 2: EXTRACT TEXT WITH NORMALIZED SIZE AND LEVEL ===
for page_num, items in ocr_results.items():
    if page_num - 1 >= len(doc): continue
    page = doc[page_num - 1]
    pix = page.get_pixmap(matrix=fitz.Matrix(DPI / 72, DPI / 72))
    img_width, img_height = pix.width, pix.height
    pdf_width, pdf_height = page.rect.width, page.rect.height

    for item in items:
        x1, y1, x2, y2 = item["box"]
        label_type = item.get("type", "").lower()
        if label_type != "heading":
            continue

        w_pad = int((x2 - x1) * PADDING)
        h_pad = int((y2 - y1) * PADDING)
        x1_pad = max(x1 - w_pad, 0)
        y1_pad = max(y1 - h_pad, 0)
        x2_pad = min(x2 + w_pad, img_width)
        y2_pad = min(y2 + h_pad, img_height)

        rect = fitz.Rect(
            x1_pad * pdf_width / img_width,
            y1_pad * pdf_height / img_height,
            x2_pad * pdf_width / img_width,
            y2_pad * pdf_height / img_height,
        )

        spans = page.get_text("dict", clip=rect).get("blocks", [])
        extracted_text = ""
        font_sizes, normalized_font_sizes, fonts, flags = [], [], [], []
        is_bold = 0
        max_local_font = 0

        for block in spans:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    size = round(span.get("size", 0), 2)
                    font = span.get("font", "")
                    flag = span.get("flags", 0)

                    extracted_text += text
                    font_sizes.append(size)
                    fonts.append(font)
                    flags.append(flag)
                    max_local_font = max(max_local_font, size)

                    if "bold" in font.lower():
                        is_bold = 1
                    elif (flag & 2) and not any(x in font for x in ["MI", "SY", "CMMI", "CMSY", "CMEX", "Symbol", "Math"]):
                        is_bold = 1

        if not font_sizes or max_local_font == 0:
            continue

        norm = round(max_local_font / max_heading_font_size, 3)
        if norm >= 0.95:
            level = "H1"
        elif norm >= 0.75:
            level = "H2"
        elif norm >= 0.50:
            level = "H3"
        else:
            continue

        training_results.append({
            "page": page_num,
            "text": extracted_text.strip(),
            "font_sizes": font_sizes,
            "normalized_font_sizes": [round(s / max_heading_font_size, 3) for s in font_sizes],
            "fonts": fonts,
            "flags": flags,
            "bbox": [x1_pad, y1_pad, x2_pad, y2_pad],
            "is_bold": is_bold,
            "level": level,
            "label": level,
            "class": label_type
        })

with open(training_output_path, "w", encoding="utf-8") as f:
    json.dump(training_results, f, indent=2, ensure_ascii=False)

print(f"✅ Training JSON with heading levels saved in: {training_output_path}")

# === PHASE 5: SERIAL HEADING CLASSIFICATION USING MULTICHANNEL LOGIC ===
# === Extract Title Text if Detected ===
title_text = ""
for item in training_results:
    if item.get("class") == "title":
        title_text = item.get("text", "").strip()
        break  # use the first detected title

print("\n🧠 Running serial heading classification using font, family and prefix logic...")

import re

CLASSIFIED_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "classified_headings.json")

def extract_prefix_level(text):
    prefix_match = re.match(r"^([0-9]+([.,][0-9]+)*)([^\w]|\s)?", text.strip())
    if prefix_match:
        prefix = prefix_match.group(1)
        if prefix.count(".") >= 2:
            return "H3"
        elif prefix.count(".") == 1:
            return "H2"
        else:
            return "H1"
    return None

def font_family_map(fonts):
    return tuple(sorted(set(f.lower().split("-")[0] for f in fonts if f)))

def match_family_familyset(family_set, font_family, font_size):
    for fam_fonts, fam_size in family_set:
        if font_family == fam_fonts and abs(fam_size - font_size) <= 0.08 * fam_size:
            return True
    return False

def get_last_fontsize(family_set):
    return max([s for _, s in family_set], default=0)

# === Load training extracted headings ===
with open(training_output_path, "r", encoding="utf-8") as f:
    extracted_headings = json.load(f)

# === Sort by page + y (top to bottom order) ===
def sort_key(h):
    return (h["page"], h["bbox"][1])

sorted_headings = sorted(extracted_headings, key=sort_key)

# === Classification Loop ===
classified = []
h1_families, h2_families, h3_families = [], [], []

for h in sorted_headings:
    text = h["text"]
    font_sizes = h["font_sizes"]
    fonts = h["fonts"]
    max_size = max(font_sizes)
    norm_size = round(max_size / max_heading_font_size, 3)
    font_family = font_family_map(fonts)
    prefix_level = extract_prefix_level(text)

    # Channel 1: Prefix Rule (overrides all)
    if prefix_level:
        level = prefix_level
    # Channel 2: Match to existing families
    elif match_family_familyset(h1_families, font_family, max_size):
        level = "H1"
    elif match_family_familyset(h2_families, font_family, max_size):
        level = "H2"
    elif match_family_familyset(h3_families, font_family, max_size):
        level = "H3"
    else:
        # Channel 3: Relative fallback (based on font size hierarchy)
        if not h1_families:
            level = "H1"
            h1_families.append((font_family, max_size))
        elif max_size > get_last_fontsize(h1_families):
            level = "H1"
            h1_families.append((font_family, max_size))
        elif not h2_families or max_size > get_last_fontsize(h2_families):
            level = "H2"
            h2_families.append((font_family, max_size))
        else:
            level = "H3"
            h3_families.append((font_family, max_size))

    h["level"] = level
    h["label"] = level
    h["normalized_font_sizes"] = [round(s / max_heading_font_size, 3) for s in font_sizes]
    classified.append(h)

# === Save results ===
with open(CLASSIFIED_OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(classified, f, indent=2, ensure_ascii=False)

print(f"✅ Final heading levels written to: {CLASSIFIED_OUTPUT_PATH}")

# === PHASE 6: FINAL PDF ANNOTATION WITH H1/H2/H3 LABELS ===
print("\n🖍️ Annotating PDF with classified heading levels...")
start_final_annot = time.time()

final_annotated_pdf = os.path.join(OUTPUT_DIR, "annotated_final_levels.pdf")
doc = fitz.open(PDF_PATH)

with open(CLASSIFIED_OUTPUT_PATH, "r", encoding="utf-8") as f:
    classified_headings = json.load(f)

for item in classified_headings:
    page_idx = item["page"] - 1
    if page_idx >= len(doc):
        continue

    page = doc[page_idx]
    x1, y1, x2, y2 = item["bbox"]
    label = item.get("level", "H?")

    # Map label to color
    color_map = {
        "H1": (1, 0, 0),   # Red
        "H2": (0, 0, 1),   # Blue
        "H3": (0, 0.5, 0)  # Green
    }
    color = color_map.get(label, (0, 0, 0))  # default black

    pix = page.get_pixmap(matrix=fitz.Matrix(DPI / 72, DPI / 72))
    img_width, img_height = pix.width, pix.height
    pdf_width, pdf_height = page.rect.width, page.rect.height

    rect = fitz.Rect(
        x1 * pdf_width / img_width,
        y1 * pdf_height / img_height,
        x2 * pdf_width / img_width,
        y2 * pdf_height / img_height,
    )

    # Draw box and insert label
    page.draw_rect(rect, color=color, width=1)
    page.insert_textbox(rect, label, fontsize=7, color=color)

# Save final annotated PDF
doc.save(final_annotated_pdf)
doc.close()

print(f"✅ Final annotated PDF with heading levels saved at: {final_annotated_pdf}")
print(f"✅ Final annotation done in {time.time() - start_final_annot:.2f}s")



# === PHASE 6: GENERATE FINAL OUTPUT.JSON ===
print("\n📝 Generating final output structure...")

final_output = {
    "title": title_text,  # Use detected title or keep blank
    "outline": []
}

for item in classified:
    level = item["level"]
    text = item["text"]
    page = item["page"]

    final_output["outline"].append({
        "level": level,
        "text": text.strip(),
        "page": page
    })

# Sort outline by page and position (already done earlier)
output_path = os.path.join(out, "output.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(final_output, f, indent=2, ensure_ascii=False)

print(f"✅ Final output saved at: {output_path}")

print(f"\n⏱️ Total pipeline completed in {time.time() - total_start:.2f} seconds.")