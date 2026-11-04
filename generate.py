#!/usr/bin/env python3
"""
Synthetic handwritten After Visit Summary generator.

  outline.yaml    -> document structure + sentence templates (Jinja2)
  diagnoses.yaml  -> medical knowledge base (codes, meds, rules)
  style.yaml      -> handwriting + layout + scan parameters measured from the real scan
  cases.yaml      -> which diagnoses each page contains

Output (in --out):
  images/<id>.png         black-and-white "scanned" page
  labels/<id>.json        full text, line polygons, diagnoses, meds, sampled style
  det_gt.txt              PaddleOCR detection labels
  rec/ + rec_gt.txt       line crops + PaddleOCR recognition labels (with --rec-crops)

Usage:
  python generate.py --outline outline.yaml --style style.yaml --cases cases.yaml --out out
  python generate.py ... --time-budget 600     # stop after 10 minutes, keep what is done
  python generate.py ... --limit 20            # quick test / timing run
"""
import argparse
import datetime as dt
import json
import math
import os
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import jinja2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont


# =====================================================================
# Sampling helpers
# =====================================================================
def sample(spec, rng):
    """12 -> 12;  {mean, sd} -> clipped normal;  [lo, hi] -> uniform."""
    if spec is None:
        return None
    if isinstance(spec, (int, float)):
        return float(spec)
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        return rng.uniform(float(spec[0]), float(spec[1]))
    if isinstance(spec, dict) and "mean" in spec:
        m, sd = float(spec["mean"]), float(spec.get("sd", 0))
        v = rng.gauss(m, sd) if sd > 0 else m
        return min(max(v, m - 3 * sd), m + 3 * sd)
    raise ValueError(f"Cannot sample from {spec!r}")


def join_and(items):
    items = [str(i) for i in items if str(i).strip()]
    if len(items) <= 1:
        return "".join(items)
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


class Overflow(Exception):
    pass


# =====================================================================
# Content: knowledge base + templates -> text blocks
# =====================================================================
class ContentBuilder:
    def __init__(self, outline, kb):
        self.outline = outline
        self.kb = kb
        self.rng = random.Random(0)
        env = jinja2.Environment(undefined=jinja2.StrictUndefined, autoescape=False)
        env.filters["pick"] = lambda xs: self.rng.choice(list(xs)) if xs else ""
        env.filters["sample"] = lambda xs, k: self.rng.sample(list(xs), min(int(k), len(xs)))
        env.filters["join_and"] = join_and
        env.filters["cap"] = lambda s: s[:1].upper() + s[1:] if s else s
        self.env = env
        self._tpl, self._expr = {}, {}

    def _render(self, src, ctx):
        if src not in self._tpl:
            self._tpl[src] = self.env.from_string(src)
        return " ".join(self._tpl[src].render(**ctx).split())

    def _when(self, expr, ctx):
        if expr not in self._expr:
            self._expr[expr] = self.env.compile_expression(expr, undefined_to_none=True)
        return bool(self._expr[expr](**ctx))

    # ---- medical rules ------------------------------------------------
    def resolve_medications(self, codes):
        present = set(codes)
        meds, seen, rules = [], set(), []
        for code in codes:
            for m in self.kb[code].get("medications") or []:
                med = {k: m[k] for k in ("name", "dose", "freq", "purpose")}
                med["note"] = ""
                conflicts = [c for c in (m.get("avoid_if") or []) if c in present]
                if conflicts:
                    why = self.kb[conflicts[0]]["plain"]
                    alt = m.get("alternative")
                    rules.append(f"{m['name']} removed ({code}) because of {conflicts[0]}"
                                 + (f"; replaced by {alt['name']}" if alt else ""))
                    if not alt:
                        continue
                    med = {k: alt[k] for k in ("name", "dose", "freq", "purpose")}
                    med["note"] = f"instead of {m['name']} because of your {why}"
                for c, dose in (m.get("dose_if") or {}).items():
                    if c in present:
                        rules.append(f"{m['name']} dose {m['dose']} -> {dose} because of {c}")
                        med["dose"] = dose
                        med["note"] = f"lower dose because of your {self.kb[c]['plain']}"
                if med["name"].lower() in seen:
                    continue
                seen.add(med["name"].lower())
                med["from_code"] = code
                meds.append(med)
        return meds, rules

    def fake_patient(self):
        fd, r = self.outline.get("fake_data", {}), self.rng

        def rand_date(y0, y1):
            d = dt.date(r.randint(y0, y1), 1, 1) + dt.timedelta(days=r.randint(0, 364))
            if r.random() < 0.5:
                return f"{d.month:02d}/{d.day:02d}/{d.year}"
            return f"{d.month}/{d.day}/{d.year % 100:02d}"

        first, last = r.choice(fd["first_names"]), r.choice(fd["last_names"])
        return {
            "patient": {"first": first, "name": f"{first} {last}",
                        "dob": rand_date(*fd.get("birth_year_range", [1940, 2004]))},
            "visit_date": rand_date(*fd.get("visit_year_range", [2024, 2026])),
            "provider": r.choice(fd["providers"]),
        }

    # ---- build the page's text blocks -----------------------------------
    def build(self, codes, seed):
        self.rng.seed(seed)
        r = self.rng
        present = set(codes)
        diagnoses = []
        for c in codes:
            e = self.kb[c]
            extra = []
            for other, texts in (e.get("extra_instructions_if") or {}).items():
                if other in present:
                    extra += list(texts)
            diagnoses.append({
                "code": c, "name": e["name"], "plain": e["plain"],
                "instructions": list(e.get("instructions") or []),
                "warning_signs": list(e.get("warning_signs") or []),
                "follow_up_weeks": e.get("follow_up_weeks"), "extra": extra,
            })
        meds, rules = self.resolve_medications(codes)
        weeks = [d["follow_up_weeks"] for d in diagnoses if d["follow_up_weeks"]]
        ctx = dict(self.fake_patient(), diagnoses=diagnoses, medications=meds,
                   follow_up_weeks=min(weeks) if weeks else 4,
                   dx_plain_list=join_and(d["plain"] for d in diagnoses),
                   dx_name_list=join_and(d["name"] for d in diagnoses))

        blocks = []  # each: {kind: heading|para|item, text, style, prefix}
        for sec in self.outline["sections"]:
            if r.random() > sec.get("p", 1.0):
                continue
            style = sec.get("style", "handwritten")
            fmt = sec.get("format", "paragraph")
            heading = r.choice(sec["heading"]) if sec.get("heading") else None
            rep = sec.get("repeat")
            items = {"diagnoses": diagnoses, "medications": meds}.get(rep, [None]) if rep else [None]
            if rep and not items:
                continue
            bullet = r.choice(sec.get("bullets", ["-"]))
            texts = []
            for n, item in enumerate(items, 1):
                lctx = dict(ctx, n=n)
                if rep == "diagnoses":
                    lctx["dx"] = item
                elif rep == "medications":
                    lctx["med"] = item
                parts = []
                for s in sec["sentences"]:
                    s = s if isinstance(s, dict) else {"variants": s}
                    if "when" in s and not self._when(s["when"], lctx):
                        continue
                    if r.random() > s.get("p", 1.0):
                        continue
                    t = self._render(r.choice(s["variants"]), lctx)
                    if t:
                        parts.append(t)
                if parts:
                    texts.append((n, " ".join(parts)))
                    if fmt == "lines":
                        texts[-1] = (n, parts)
            if not texts:
                continue
            if fmt == "lines":  # every sentence on its own line
                if heading:
                    blocks.append({"kind": "heading", "text": heading, "style": style})
                for _, parts in texts:
                    for t in parts:
                        blocks.append({"kind": "item", "text": t, "style": style})
            elif fmt == "list":
                if heading:
                    blocks.append({"kind": "heading", "text": heading, "style": style})
                for n, t in texts:
                    b = bullet.replace("{n}", str(n)).strip()
                    blocks.append({"kind": "item", "text": (b + " " + t).strip(), "style": style})
            else:
                body = " ".join(t for _, t in texts)
                if heading and sec.get("heading_inline"):
                    blocks.append({"kind": "para", "text": heading + " " + body, "style": style})
                else:
                    if heading:
                        blocks.append({"kind": "heading", "text": heading, "style": style})
                    blocks.append({"kind": "para", "text": body, "style": style})
            blocks.append({"kind": "gap"})

        meta = {"diagnoses": [{"code": d["code"], "name": d["name"]} for d in diagnoses],
                "medications": [{k: m[k] for k in ("name", "dose", "freq", "from_code")} for m in meds],
                "rules_applied": rules, "patient": ctx["patient"], "visit_date": ctx["visit_date"],
                "provider": ctx["provider"], "follow_up_weeks": ctx["follow_up_weeks"]}
        return blocks, meta


# =====================================================================
# Rendering: per-character jitter handwriting
# =====================================================================
class Renderer:
    def __init__(self, style, base_dir):
        self.s = style
        self.W = int(style["page"]["width_in"] * style["dpi"])
        self.H = int(style["page"]["height_in"] * style["dpi"])
        fdir = (base_dir / style.get("fonts_dir", "fonts")).resolve()
        self.hand = []
        for fname, w in style["handwriting_fonts"].items():
            if not w:
                continue
            p = fdir / fname
            if not p.exists():
                raise FileNotFoundError(f"Font not found: {p}")
            self.hand.append((str(p), float(w), self._xh_ratio(str(p))))
        self.printed = None
        if style.get("printed_font"):
            p = str(fdir / style["printed_font"])
            self.printed = (p, 1.0, self._xh_ratio(p))
        self._fonts, self._glyphs = {}, {}

    @staticmethod
    def _xh_ratio(path):
        f = ImageFont.truetype(path, 200)
        l, t, r, b = f.getbbox("x", anchor="ls")
        return max(b - t, 1) / 200.0

    def font(self, path, size):
        k = (path, size)
        if k not in self._fonts:
            self._fonts[k] = ImageFont.truetype(path, size)
        return self._fonts[k]

    def glyph(self, path, size, ch):
        """Returns (mask float32 0..1, left, top) relative to the baseline origin, and advance."""
        k = (path, size, ch)
        if k not in self._glyphs:
            f = self.font(path, size)
            adv = f.getlength(ch)
            l, t, r, b = f.getbbox(ch, anchor="ls")
            if r - l <= 0 or b - t <= 0:
                self._glyphs[k] = (None, 0, 0, adv)
            else:
                im = Image.new("L", (r - l + 2, b - t + 2), 0)
                ImageDraw.Draw(im).text((1 - l, 1 - t), ch, font=f, fill=255, anchor="ls")
                self._glyphs[k] = (im, l - 1, t - 1, adv)
        return self._glyphs[k]

    # ---- page-level parameters ----------------------------------------
    def page_params(self, rng, scale):
        s = self.s
        paths = self.hand
        font = rng.choices(paths, weights=[w for _, w, _ in paths])[0]
        xh = sample(s["x_height_px"], rng) * scale
        return {
            "font": font, "x_height": xh,
            "size": max(8, int(round(xh / font[2]))),
            "line_spacing": sample(s["line_spacing_px"], rng) * scale,
            "slant": sample(s["slant_deg"], rng),
            "margin_l": sample(s["margins_px"]["left"], rng),
            "margin_t": sample(s["margins_px"]["top"], rng),
            "margin_b": sample(s["margins_px"]["bottom"], rng),
            "margin_r": sample(s["margins_px"]["right"], rng),
            "indent": sample(s["paragraph"]["indent_px"], rng),
            "stroke": sample(s["stroke_width_px"], rng),
            "ink": sample(s["ink_density"], rng),
            "scale": scale,
        }

    # ---- layout words into lines --------------------------------------
    def layout(self, blocks, P, rng):
        s, cj = self.s, self.s["char_jitter"]
        lines = []  # each: dict(words=[(x, plans)], text, baseline, style, kind)
        y = P["margin_t"] + P["line_spacing"] * 0.8
        bottom = self.H - P["margin_b"]
        for blk in blocks:
            if blk["kind"] == "gap":
                y += P["line_spacing"] * s["paragraph"].get("gap_lines", 0.5)
                continue
            printed = blk["style"] == "printed" and self.printed
            fpath, _, ratio = self.printed if printed else P["font"]
            xh = P["x_height"] * (s.get("heading_scale", 1.0) if blk["kind"] == "heading" else 1.0)
            base_size = max(8, int(round(xh / ratio)))
            x0 = P["margin_l"]
            first_x = x0 + (P["indent"] if blk["kind"] == "para" else 0)
            cont_x = x0 + (s.get("list_item_indent_px", 0) if blk["kind"] == "item" else 0)
            words = blk["text"].split()
            cur, x = [], first_x
            limit = self.W - P["margin_r"] - abs(sample(s["right_ragged_px"], rng))

            def flush():
                nonlocal y
                if y > bottom:
                    raise Overflow()
                lines.append({"words": list(cur), "text": " ".join(w for w, _, _ in cur),
                              "baseline": y, "printed": bool(printed), "kind": blk["kind"]})
                y += P["line_spacing"] + sample(s["line_spacing_jitter_px"], rng)

            for w in words:
                plans, wx = [], 0.0
                for ch in w:
                    if printed:
                        size, rot, dy, sp = base_size, 0.0, 0.0, 0.0
                    else:
                        size = max(6, int(round(base_size * (1 + sample(cj["size_rel"], rng)))))
                        rot = sample(cj["rotation_deg"], rng)
                        dy = sample(cj["baseline_px"], rng)
                        sp = sample(cj["spacing_px"], rng)
                    adv = self.glyph(fpath, size, ch)[3]
                    plans.append((ch, fpath, size, rot, dy, wx))
                    wx += adv + sp
                if cur and x + wx > limit:
                    flush()
                    cur, x = [], cont_x
                    limit = self.W - P["margin_r"] - abs(sample(s["right_ragged_px"], rng))
                cur.append((w, x, plans))
                x += wx + max(4.0, sample(s["word_gap_px"], rng) * P["scale"])
            if cur:
                flush()
        return lines

    # ---- draw one line into a strip, then slant + tilt it -------------
    def draw_line(self, line, P, rng, ink):
        s, cj = self.s, self.s["char_jitter"]
        ls = P["line_spacing"]
        sh = int(ls * 2.4)
        by = int(ls * 1.4)  # baseline row inside the strip
        strip = np.zeros((sh, self.W), np.float32)
        amp = 0 if line["printed"] else sample(cj["wobble"]["amplitude_px"], rng)
        lam = sample(cj["wobble"]["wavelength_px"], rng)
        phase = rng.uniform(0, 2 * math.pi)
        for w, wx0, plans in line["words"]:
            for ch, fpath, size, rot, dy, off in plans:
                im, l, t, _ = self.glyph(fpath, size, ch)
                if im is None:
                    continue
                gx = wx0 + off
                yb = by + dy + amp * math.sin(2 * math.pi * gx / lam + phase)
                cx, cy = gx + l + im.width / 2, yb + t + im.height / 2
                g = im.rotate(rot, resample=Image.BILINEAR, expand=True) if abs(rot) > 0.2 else im
                a = np.asarray(g, np.float32) / 255.0
                x1, y1 = int(round(cx - a.shape[1] / 2)), int(round(cy - a.shape[0] / 2))
                xa, ya = max(0, x1), max(0, y1)
                xb, yb2 = min(self.W, x1 + a.shape[1]), min(sh, y1 + a.shape[0])
                if xb <= xa or yb2 <= ya:
                    continue
                sub = a[ya - y1:yb2 - y1, xa - x1:xb - x1]
                np.maximum(strip[ya:yb2, xa:xb], sub, out=strip[ya:yb2, xa:xb])

        # slant (shear about the baseline) + baseline tilt (rotation about line start)
        slant = 0.0 if line["printed"] else P["slant"]
        tilt = sample(s["baseline_tilt_deg"], rng)
        px = P["margin_l"]
        sh_m = np.array([[1, math.tan(math.radians(slant)), -math.tan(math.radians(slant)) * by],
                         [0, 1, 0], [0, 0, 1]], np.float64)
        rot_m = np.vstack([cv2.getRotationMatrix2D((px, by), tilt, 1.0), [0, 0, 1]])
        M = (rot_m @ sh_m)[:2]
        warped = cv2.warpAffine(strip, M, (self.W, sh), flags=cv2.INTER_LINEAR, borderValue=0)

        top = int(round(line["baseline"])) - by
        ya, yb = max(0, top), min(self.H, top + sh)
        if yb > ya:
            np.maximum(ink[ya:yb], warped[ya - top:yb - top], out=ink[ya:yb])
        ys, xs = np.nonzero(warped > 0.3)
        if len(xs) == 0:
            return None
        return [float(xs.min()), float(ys.min() + top), float(xs.max()), float(ys.max() + top)]

    # ---- elastic distortion: bends strokes so fonts look less regular ---
    def elastic(self, ink, rng, nrng):
        e = self.s.get("elastic")
        if not e:
            return ink
        alpha, sigma = sample(e["alpha_px"], rng), sample(e["sigma_px"], rng)
        if not alpha:
            return ink
        h, w = ink.shape
        sm = max(1, int(round(sigma / 4)))          # build the field at low res for speed
        hs, ws = h // sm + 1, w // sm + 1
        dx = cv2.GaussianBlur(nrng.uniform(-1, 1, (hs, ws)).astype(np.float32), (0, 0), sigma / sm)
        dy = cv2.GaussianBlur(nrng.uniform(-1, 1, (hs, ws)).astype(np.float32), (0, 0), sigma / sm)
        for d in (dx, dy):
            d *= alpha / (np.abs(d).max() + 1e-6)
        dx = cv2.resize(dx, (w, h))
        dy = cv2.resize(dy, (w, h))
        xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        return cv2.remap(ink, xx + dx, yy + dy, cv2.INTER_LINEAR, borderValue=0)

    # ---- pen width matching -------------------------------------------
    @staticmethod
    def match_stroke(ink, target):
        b = (ink > 0.5).astype(np.uint8)
        area = int(b.sum())
        if area == 0:
            return ink, 0.0
        cnts, _ = cv2.findContours(b, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        per = sum(cv2.arcLength(c, True) for c in cnts)
        cur = 2.0 * area / max(per, 1.0)
        k = int(round(target - cur))
        if k > 0:
            if k >= 2:
                ink = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=k // 2)
            if k % 2:
                ink = cv2.dilate(ink, np.ones((2, 2), np.uint8))
        elif k < 0:
            k = min(-k, int(cur - 1.5))  # never erode strokes away
            if k >= 2:
                ink = cv2.erode(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=k // 2)
            if k % 2:
                ink = cv2.erode(ink, np.ones((2, 2), np.uint8))
        return ink, cur

    # ---- scanner simulation -------------------------------------------
    def scan(self, ink, P, rng, nrng, boxes):
        sc = self.s["scan"]
        ink = ink * P["ink"]
        sig = sample(sc["blur_sigma"], rng)
        if sig and sig > 0:
            ink = cv2.GaussianBlur(ink, (0, 0), sig)
        gray = 1.0 - ink + nrng.normal(0, sample(sc["paper_noise_sd"], rng), ink.shape).astype(np.float32)
        gray = np.clip(gray, 0, 1)

        skew = sample(sc["skew_deg"], rng)
        M = cv2.getRotationMatrix2D((self.W / 2, self.H / 2), skew, 1.0)
        gray = cv2.warpAffine(gray, M, (self.W, self.H), flags=cv2.INTER_LINEAR, borderValue=1.0)
        polys = []
        for (x0, y0, x1, y1) in boxes:
            pad = 3
            pts = np.array([[x0 - pad, y0 - pad], [x1 + pad, y0 - pad], [x1 + pad, y1 + pad], [x0 - pad, y1 + pad]])
            pts = (np.hstack([pts, np.ones((4, 1))]) @ M.T)
            pts[:, 0] = np.clip(pts[:, 0], 0, self.W - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, self.H - 1)
            polys.append([[int(round(a)), int(round(b))] for a, b in pts])

        g8 = (gray * 255).astype(np.uint8)
        mode = sc.get("binarize", "otsu")
        if mode == "otsu":
            _, out = cv2.threshold(g8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif mode == "adaptive":
            out = cv2.adaptiveThreshold(g8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                        int(sc.get("adaptive_block", 51)), float(sc.get("adaptive_c", 12)))
        elif mode == "fixed":
            _, out = cv2.threshold(g8, int(255 * float(sc.get("fixed_threshold", 0.5))), 255, cv2.THRESH_BINARY)
        else:
            out = g8

        # broken strokes: small white gaps inside ink
        rate = sample(sc["stroke_break_rate"], rng)
        if rate and rate > 0:
            ys, xs = np.nonzero(out < 128)
            n = int(len(xs) * rate)
            if n:
                idx = nrng.choice(len(xs), n, replace=False)
                for i in idx:
                    cv2.circle(out, (int(xs[i]), int(ys[i])), int(nrng.integers(1, 3)), 255, -1)
        # speckle: isolated black dots
        rate = sample(sc["speckle_rate"], rng)
        if rate and rate > 0:
            n = int(self.W * self.H * rate)
            out[nrng.integers(0, self.H, n), nrng.integers(0, self.W, n)] = 0
        q = sc.get("jpeg_quality")
        if q:
            ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
            out = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
            if mode in ("otsu", "adaptive", "fixed"):
                out = np.where(out < 128, 0, 255).astype(np.uint8)
        return out, polys, skew

    def render(self, blocks, P, rng, nrng):
        lines = self.layout(blocks, P, rng)
        ink = np.zeros((self.H, self.W), np.float32)
        kept, boxes = [], []
        for ln in lines:
            box = self.draw_line(ln, P, rng, ink)
            if box:
                kept.append(ln)
                boxes.append(box)
        ink = self.elastic(ink, rng, nrng)
        ink, measured = self.match_stroke(ink, P["stroke"])
        img, polys, skew = self.scan(ink, P, rng, nrng, boxes)
        return img, kept, polys, {"stroke_before_match": round(measured, 2), "skew_deg": round(skew, 2)}


# =====================================================================
# Worker
# =====================================================================
W = {}


def init_worker(outline_p, style_p, out_dir, rec_crops):
    outline = yaml.safe_load(open(outline_p))
    style = yaml.safe_load(open(style_p))
    kb_path = Path(outline_p).parent / outline["knowledge_base"]
    kb = yaml.safe_load(open(kb_path))
    W.update(builder=ContentBuilder(outline, kb), renderer=Renderer(style, Path(style_p).parent),
             outline=outline, kb=kb, out=Path(out_dir), rec=rec_crops)


def mentions_all(text, codes, kb):
    low = text.lower()
    return all(c.lower() in low or kb[c]["name"].lower() in low or kb[c]["plain"].lower() in low
               for c in codes)


def make_page(task):
    page_id, case_id, codes, seed = task
    t0 = time.time()
    b, R, kb = W["builder"], W["renderer"], W["kb"]
    val = W["outline"].get("validation", {})
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    last_err = "unknown"
    for attempt in range(int(val.get("max_retries", 4)) + 1):
        blocks, meta = b.build(codes, seed * 31 + attempt)
        P = R.page_params(rng, scale=0.9 ** attempt)
        try:
            img, lines, polys, extra = R.render(blocks, P, rng, nrng)
        except Overflow:
            last_err = "text overflowed the page"
            continue
        full = "\n".join(l["text"] for l in lines)
        if val.get("require_each_diagnosis", True) and not mentions_all(full, codes, kb):
            last_err = "a diagnosis was not mentioned"
            continue
        break
    else:
        return {"id": page_id, "error": last_err}

    out = W["out"]
    img_rel = f"images/{page_id}.png"
    pil = Image.fromarray(img)
    if R.s["scan"].get("output_mode", "binary") == "binary":
        pil = pil.convert("1")
    pil.save(out / img_rel, optimize=False)

    line_labels = [{"text": l["text"], "kind": l["kind"], "points": p} for l, p in zip(lines, polys)]
    rec = []
    if W["rec"]:
        for i, (l, p) in enumerate(zip(lines, polys)):
            xs, ys = [q[0] for q in p], [q[1] for q in p]
            crop = img[max(0, min(ys)):max(ys) + 1, max(0, min(xs)):max(xs) + 1]
            rel = f"rec/{page_id}_{i:02d}.png"
            cv2.imwrite(str(out / rel), crop)
            rec.append((rel, l["text"]))

    label = {
        "id": page_id, "case_id": case_id, "seed": seed, "image": img_rel,
        **meta, "text": full, "lines": line_labels,
        "style_sampled": {"font": Path(P["font"][0]).name, "x_height_px": round(P["x_height"], 1),
                          "font_size": P["size"], "line_spacing_px": round(P["line_spacing"], 1),
                          "slant_deg": round(P["slant"], 2), "stroke_target_px": round(P["stroke"], 2),
                          "retries": attempt, **extra},
    }
    with open(out / f"labels/{page_id}.json", "w") as f:
        json.dump(label, f, indent=1, ensure_ascii=False)
    det = [{"transcription": l["text"], "points": l["points"]} for l in line_labels]
    return {"id": page_id, "det": f"{img_rel}\t{json.dumps(det, ensure_ascii=False)}",
            "rec": rec, "secs": time.time() - t0}


# =====================================================================
# Main
# =====================================================================
def build_tasks(cases_cfg, kb):
    rng = random.Random(cases_cfg.get("seed", 0))
    base = cases_cfg.get("seed", 0) * 1_000_003
    tasks = []
    for ci, c in enumerate(cases_cfg.get("cases") or []):
        codes = list(c["diagnoses"])
        bad = [x for x in codes if x not in kb]
        if bad:
            sys.exit(f"Case {c.get('id', ci)}: codes not in knowledge base: {bad}")
        cid = c.get("id", f"case{ci}")
        for k in range(int(c.get("copies", 1))):
            tasks.append((f"{cid}_{k:04d}", cid, codes))
    rnd = cases_cfg.get("random")
    if rnd and rnd.get("n"):
        pool = list(kb) if rnd.get("pool", "all") == "all" else list(rnd["pool"])
        lo, hi = rnd.get("diagnoses_per_case", [1, 3])
        for k in range(int(rnd["n"])):
            codes = rng.sample(pool, rng.randint(lo, min(hi, len(pool))))
            tasks.append((f"random_{k:05d}", "random", codes))
    return [(pid, cid, codes, base + i) for i, (pid, cid, codes) in enumerate(tasks)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outline", default="outline.yaml")
    ap.add_argument("--style", default="style.yaml")
    ap.add_argument("--cases", default="cases.yaml")
    ap.add_argument("--out", default="out")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=None, help="only generate the first N pages")
    ap.add_argument("--time-budget", type=float, default=None, help="seconds; stop and keep finished pages")
    ap.add_argument("--rec-crops", action="store_true", help="also write line crops for recognizer training")
    a = ap.parse_args()

    outline = yaml.safe_load(open(a.outline))
    kb = yaml.safe_load(open(Path(a.outline).parent / outline["knowledge_base"]))
    tasks = build_tasks(yaml.safe_load(open(a.cases)), kb)
    if a.limit:
        tasks = tasks[:a.limit]
    out = Path(a.out)
    for d in ("images", "labels") + (("rec",) if a.rec_crops else ()):
        (out / d).mkdir(parents=True, exist_ok=True)

    print(f"{len(tasks)} pages queued, {a.workers} workers"
          + (f", time budget {a.time_budget:.0f}s" if a.time_budget else ""), flush=True)
    t0 = time.time()
    det, rec, errors, done = [], [], [], 0
    pool = Pool(a.workers, initializer=init_worker,
                initargs=(a.outline, a.style, str(out), a.rec_crops))
    stopped = False
    try:
        for res in pool.imap_unordered(make_page, tasks, chunksize=2):
            if "error" in res:
                errors.append(res)
            else:
                done += 1
                det.append(res["det"])
                rec += res["rec"]
            el = time.time() - t0
            if done and done % 25 == 0:
                rate = done / el
                print(f"  {done}/{len(tasks)} pages  {rate:.2f} pages/s  "
                      f"~{(len(tasks) - done) / rate:.0f}s left", flush=True)
            if a.time_budget and el > a.time_budget:
                stopped = True
                break
    finally:
        if stopped:
            pool.terminate()
        else:
            pool.close()
        pool.join()

    det.sort()
    (out / "det_gt.txt").write_text("\n".join(det) + "\n", encoding="utf-8")
    if a.rec_crops:
        (out / "rec_gt.txt").write_text("\n".join(f"{p}\t{t}" for p, t in sorted(rec)) + "\n", encoding="utf-8")
    el = time.time() - t0
    print(f"Done: {done} pages in {el:.1f}s ({done / max(el, 1e-9):.2f} pages/s)"
          + (" [stopped at time budget]" if stopped else ""))
    if errors:
        print(f"{len(errors)} pages skipped:", *[f"  {e['id']}: {e['error']}" for e in errors[:10]], sep="\n")
    if done:
        print(f"At this rate, 10 minutes = about {int(600 * done / el)} pages with {a.workers} workers.")


if __name__ == "__main__":
    main()
