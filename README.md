# Synthetic handwritten After Visit Summary generator

Generates black-and-white "scanned" pages of handwritten after-visit summaries containing the
diagnoses you specify, with exact ground truth for OCR and ICD-10 evaluation.

## Files

| File | What you edit |
|---|---|
| `style.yaml` | Measurements from your real scan: margins, line spacing, x-height, slant, stroke width, per-character jitter, fonts, scan noise |
| `outline.yaml` | Section order, headings, and sentence templates (Jinja2) |
| `diagnoses.yaml` | Medical knowledge base: ICD-10-CM codes, meds, instructions, warning signs, and rules (`avoid_if`, `alternative`, `dose_if`, `extra_instructions_if`) |
| `cases.yaml` | Which diagnoses each page contains and how many copies, plus optional random combinations |
| `fonts/` | Open-licensed (OFL) handwriting fonts from Google Fonts. Add your own `.ttf` files and list them in `style.yaml` |

## Run

```bash
pip install pillow numpy opencv-python-headless pyyaml jinja2
python generate.py --outline outline.yaml --style style.yaml --cases cases.yaml --out out --rec-crops
python generate.py ... --time-budget 600   # stop at 10 minutes, keep finished pages
python generate.py ... --limit 20          # quick timing test
```

On 2 CPU cores this ran at about 4 pages/s, roughly 2,400 pages in 10 minutes. It scales with `--workers`.

## Output

- `images/<id>.png`: 300 DPI 1-bit page
- `labels/<id>.json`: full text, line polygons, diagnoses, medications after the rules were applied, which rules fired, and the style values sampled for that page
- `det_gt.txt`: PaddleOCR detection labels
- `rec/` + `rec_gt.txt`: line crops and PaddleOCR recognition labels (with `--rec-crops`)

## Medical rules

The generator applies the rules in `diagnoses.yaml`. For example, with CKD (N18.30), ibuprofen is
replaced by acetaminophen, nitrofurantoin by cephalexin, and metformin drops to 500 mg. Templates
can only print values from the knowledge base. The included knowledge base is example data: have
a clinician review it before you rely on it.

## Matching your real scan

Change the values in `style.yaml` to your measurements, generate a few pages with `--limit 10`, and
compare them side by side with the real scan. Check `style_sampled` in the label files to see the
values each page used. Pick the 3 to 5 fonts closest to the real writing, and remove the others or
give them a weight of 0.
