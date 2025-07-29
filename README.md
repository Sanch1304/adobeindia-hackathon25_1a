# Challenge 1a: Heading Detection and Classification from PDF

## 🧠 Objective

The objective of Challenge 1a is to detect and classify heading elements (titles, headings, and their levels H1/H2/H3) in complex PDFs and generate a structured JSON output representing the document's outline.

---

## 🔹 Solution Overview

This solution is designed to be accurate, modular, and Docker-compatible. It follows a six-phase pipeline to process the PDF and extract a structured outline of the headings:

### Phase 1: PDF to Binarized Images

* Convert each PDF page to grayscale, binarized PNG images using `PyMuPDF` and `OpenCV`.
* Improves visual clarity for downstream detection.

### Phase 2: YOLOv10n Inference

* Perform object detection using a lightweight YOLOv10n model exported to ONNX.
* Only `heading` and `title` classes are retained.
* Additional filtering ensures noisy detections near images/tables are removed.

### Phase 3: Initial Annotation

* Detected heading and title boxes are overlaid on the PDF to verify bounding box accuracy.

### Phase 4: Text Extraction & Normalization

* Extract text from detected regions and collect metadata: font size, font name, font weight.
* Normalize font sizes relative to the largest heading.

### Phase 5: Heading Classification (H1/H2/H3)

* Use a multi-channel strategy:

  1. **Prefix Rule**: If the heading starts with numeric patterns like `1`, `1.1`, or `2.3.4`, it's assigned H1/H2/H3 respectively.
  2. **Font Family/Size Matching**: Match detected heading with previously seen heading families.
  3. **Fallback Logic**: If unmatched, compare font sizes hierarchically to assign H1, H2, or H3.

### Phase 6: Final JSON Output & PDF Annotation

* Outputs a structured `output.json` with a document title and heading outline.
* Saves a final annotated PDF with H1/H2/H3 boxes color-coded.

---

## 🧰 Model & Libraries Used

### Model

* Custom-trained YOLOv10n
* Format: ONNX
* Size: \~8MB
* Inference engine: `onnxruntime` (CPU-only)

### Python Libraries

* PyMuPDF (fitz)
* OpenCV (cv2)
* NumPy
* onnxruntime
* re, os, time, json, pathlib

---

## 🚜 Build & Run (Docker)

These commands are used to build and run the solution within a Docker container as per competition requirements:

### Build Docker Image

```bash
docker build --platform linux/amd64 -t headingextractor:latest .

```

### Run the Container

```bash
docker run --rm \
  -v $(pwd)/input:/workforce/input \
  -v $(pwd)/output:/workforce/output \
  --network none \
  headingextractor:latest

```

---

## 📄 Output Format (output.json)

```json
{
  "title": "Document Title",
  "outline": [
    {"level": "H1", "text": "1. Introduction", "page": 1},
    {"level": "H2", "text": "1.1 Scope", "page": 2},
    {"level": "H3", "text": "1.1.1 Background", "page": 3}
  ]
}
```

---

## ✅ Compliance Checklist

* [x] Supports AMD64 CPU architecture
* [x] Runs within 10 seconds for 50-page PDF
* [x] Model size under 200MB
* [x] No internet or GPU dependency
* [x] Fully modular and reproducible

---

## 🔎 Notes

* Detection filters out headings near image/table objects
* Uses top-down heading flow per page for correct hierarchy
* Multichannel heading classification supports edge cases
* Structure makes it easily adaptable for multilingual documents
