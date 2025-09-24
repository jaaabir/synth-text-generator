"""
Template: Whole-Word, 1-3 Lines, multi-color (restricted palette), minimum font size enforced.

- Chooses contiguous words from corpus (so no subword fragments).
- Builds 1..3 lines, each with a configurable number of words.
- Tries to fit lines in width by reducing words-per-line if needed.
- Enforces a minimum font size so text does not become tiny.
- Renders text with PIL onto a paper image (from document.paper.image.paths) and optionally pastes on background.
- Writes metadata.jsonl with text_sequence and text_colors.

Config expectations (reuse your existing config.yaml keys):
- document.paper.image.paths -> paper images directory
- document.content.text.path or paths -> corpus file(s)
- document.content.font.paths -> font directory(s)
- document.content.layout:
    text_scale: [min_frac, max_frac]  (fraction of short_size used for font size estimate)
    lines_range: [1, 3]
    words_per_line: [2, 8]   (default words-per-line range)
- min_font_px: optional top-level config key to set minimum font px (default 16)
- background.image.paths -> optional background images
"""
import os
import random
import json
import re
from typing import Any, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from synthtiger import templates
# elements.Background / Document are not required here (we do lightweight PIL composition),
# but importing Background would be fine if you prefer to use synthtiger Background.

HAVE_BACKGROUND = True



def _list_files(dirpaths, exts=None):
    out = []
    if not dirpaths:
        return out
    if isinstance(dirpaths, str):
        dirpaths = [dirpaths]
    for d in dirpaths:
        if not d:
            continue
        for root, _, files in os.walk(d):
            for f in files:
                if exts is None or os.path.splitext(f.lower())[1] in exts:
                    out.append(os.path.join(root, f))
    return out


class SynthGenerator(templates.Template):
    def __init__(self, config=None, split_ratio: List[float] = []):
        super().__init__(config)
        if config is None:
            config = {}

        # general
        self.quality = config.get("quality", [60, 95])
        self.landscape = config.get("landscape", 0.5)
        self.short_size = config.get("short_size", [720, 1024])
        self.aspect_ratio = config.get("aspect_ratio", [1, 1.6])
        self.max_rotation = config.get("max_rotation", 4)

        # minimum font size (px) -- prevents tiny text
        self.min_font_px = int(config.get("min_font_px", 16))

        # document-related config (we read needed paths directly from config)
        doc = config.get("document", {}) or {}
        paper_cfg = doc.get("paper", {}) or {}
        paper_img_cfg = paper_cfg.get("image", {}) or {}
        self.paper_dirs = paper_img_cfg.get("paths") or []
        if isinstance(self.paper_dirs, str):
            self.paper_dirs = [self.paper_dirs]

        content = doc.get("content", {}) or {}
        text_cfg = content.get("text", {}) or {}
        self.corpus_paths = text_cfg.get("path") or text_cfg.get("paths") or []
        if isinstance(self.corpus_paths, str):
            self.corpus_paths = [self.corpus_paths]

        font_cfg = content.get("font", {}) or {}
        self.font_dirs = font_cfg.get("paths") or []
        if isinstance(self.font_dirs, str):
            self.font_dirs = [self.font_dirs]

        layout = content.get("layout", {}) or {}
        self.text_scale = layout.get("text_scale", [0.03, 0.06])
        self.lines_range = layout.get("lines_range", [1, 3])
        # words per line range (controls how many words to attempt per line)
        self.words_per_line = layout.get("words_per_line", [2, 8])
        self.margin = content.get("margin", [0.05, 0.08])
        self.line_spacing = layout.get("line_spacing", 1.05)

        # background dirs (optional)
        bg_cfg = config.get("background", {}) or {}
        bg_img_cfg = bg_cfg.get("image", {}) or {}
        self.bg_dirs = bg_img_cfg.get("paths") or []
        if isinstance(self.bg_dirs, str):
            self.bg_dirs = [self.bg_dirs]

        # corpus words cache
        self._words = None

        # train/val/test split
        self.splits = ["train", "validation", "test"]
        self.split_ratio = split_ratio if split_ratio else [0.9, 0.05, 0.05]
        self.split_indexes = np.random.choice(3, size=50000, p=self.split_ratio)

        # COLOR PALETTE: only allowed colors (no gray, no light colors)
        # black, blue, green, red, dark yellow, maroon
        self._palette = [
            (0, 0, 0),        # black
            (0, 0, 180),      # blue (darker)
            (0, 128, 0),      # green (darker)
            (180, 0, 0),      # red (darker)
            (153, 128, 0),    # dark yellow / mustard
            (128, 0, 0),      # maroon
        ]

    # ---------------- helpers ----------------
    def _gather_papers(self):
        return _list_files(self.paper_dirs, exts={".jpg", ".jpeg", ".png", ".bmp", ".tif"})

    def _gather_backgrounds(self):
        return _list_files(self.bg_dirs, exts={".jpg", ".jpeg", ".png", ".bmp", ".tif"})

    def _gather_fonts(self):
        return _list_files(self.font_dirs, exts={".ttf", ".otf"})

    def _load_corpus_words(self):
        if self._words is not None:
            return
        words = []
        for p in self.corpus_paths:
            if not os.path.exists(p):
                continue
            try:
                # read full content and split into words; keep punctuation attached to words briefly
                with open(p, "r", encoding="utf-8", errors="ignore") as fp:
                    text = fp.read()
                    # normalize whitespace into single spaces
                    text = re.sub(r"\s+", " ", text).strip()
                    if text:
                        words.extend(text.split(" "))
            except Exception:
                continue
        if len(words) == 0:
            words = ["This", "is", "a", "sample", "corpus", "with", "several", "words", "for", "testing"]
        # filter any empty
        words = [w for w in words if w]
        self._words = words

    def _pick_contiguous_words(self, total_words):
        """
        Pick a contiguous span of words (length total_words) from corpus.
        If corpus shorter, we sample with wrap-around random picks.
        """
        self._load_corpus_words()
        if total_words <= 0:
            return []
        L = len(self._words)
        if L == 0:
            return []
        if L >= total_words:
            start = random.randint(0, L - total_words)
            return self._words[start : start + total_words]
        else:
            # not enough words: repeat sampling contiguous chunks and join
            out = []
            while len(out) < total_words:
                start = random.randint(0, L - 1)
                take = min(total_words - len(out), L - start)
                out.extend(self._words[start : start + take])
                if len(out) < total_words:
                    # wrap from start
                    out.extend(self._words[0 : min(L, total_words - len(out))])
            return out[:total_words]

    def _estimate_font_size(self, short_size):
        # choose a fractional font size but enforce min px
        ts_min, ts_max = (self.text_scale[0], self.text_scale[1]) if isinstance(self.text_scale, (list, tuple)) else (self.text_scale, self.text_scale)
        font_px = int(short_size * random.uniform(ts_min, ts_max))
        if font_px < self.min_font_px:
            font_px = self.min_font_px
        return font_px

    # ---------------- main API ----------------
    def generate(self):
        # image size
        landscape = random.random() < self.landscape
        short_size = random.randint(self.short_size[0], self.short_size[1])
        aspect_ratio = random.uniform(self.aspect_ratio[0], self.aspect_ratio[1])
        long_size = int(short_size * aspect_ratio)
        size = (long_size, short_size) if landscape else (short_size, long_size)
        W, H = size

        # prepare paper background (PIL)
        paper_files = self._gather_papers()
        if paper_files:
            ppath = random.choice(paper_files)
            try:
                paper = Image.open(ppath).convert("RGBA")
                paper = ImageOps.fit(paper, (W, H), Image.LANCZOS)
                
            except Exception as e:
                paper = Image.new("RGBA", (W, H), (255, 255, 255, 255))
        else:
            paper = Image.new("RGBA", (W, H), (255, 255, 255, 255))

        # optional background - paste paper over a background image if available
        bg_files = self._gather_backgrounds()
        if bg_files:
            bpath = random.choice(bg_files)
            # print(bpath)
            try:
                bg = Image.open(bpath).convert("RGBA")
                bg = ImageOps.fit(bg, (W, H), Image.LANCZOS)
                # print('Got the background')
            except Exception as e:
                print(e)
                print('generating a blank bg')
                bg = Image.new("RGBA", (W, H), (255, 255, 255, 255))
            canvas = bg.copy()
            canvas.paste(paper, (0, 0), paper)
        else:
            canvas = paper.copy()

        draw = ImageDraw.Draw(canvas)

        # choose number of lines (1..3) and words per line distribution
        n_lines = random.randint(int(self.lines_range[0]), int(self.lines_range[1]))
        n_lines = max(1, min(3, n_lines))

        wmin, wmax = (self.words_per_line[0], self.words_per_line[1]) if isinstance(self.words_per_line, (list, tuple)) else (self.words_per_line, self.words_per_line)
        wmin = max(1, int(wmin))
        wmax = max(wmin, int(wmax))
        words_per_line_choices = [random.randint(wmin, wmax) for _ in range(n_lines)]
        total_words_needed = sum(words_per_line_choices)

        words_block = self._pick_contiguous_words(total_words_needed)
        lines = []
        idx = 0
        for num in words_per_line_choices:
            chunk = words_block[idx : idx + num]
            idx += num
            line = " ".join(chunk).strip()
            lines.append(line)
            print(line)

        font_files = self._gather_fonts()
        font_px = self._estimate_font_size(short_size)
        try:
            if font_files:
                font_path = random.choice(font_files)
                font = ImageFont.truetype(font_path, max(8, font_px))
            else:
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        horiz = random.choice(["left", "center", "right"])
        vert = random.choice(["top", "center", "bottom"])

        margin_px = int(short_size * random.uniform(self.margin[0], self.margin[1])) if isinstance(self.margin, (list, tuple)) else int(short_size * float(self.margin))
        max_text_width = W - 2 * margin_px

        # Prepare sizes for center/bottom placement
        line_sizes = [draw.textsize(line, font=font) for line in lines]
        line_heights = [h for _, h in line_sizes]
        total_block_height = int(sum(line_heights) + (len(line_heights) - 1) * font.size * (self.line_spacing - 1.0))
        x = margin_px
        if vert == "top":
            y = margin_px
        elif vert == "center":
            y = (H - total_block_height) // 2
        else:  # bottom
            y = H - margin_px - (total_block_height*2)
            if y < margin_px:
                y = margin_px  # safeguard

        rendered_lines = []
        rendered_colors = []

        # FINAL: Strict width and height checks — never draw outside
        for i, line in enumerate(lines):
            cur_line = line
            words = cur_line.split(" ")
            while True:
                w, h = draw.textsize(cur_line, font=font)
                if w <= max_text_width:
                    break
                if len(words) <= 1:
                    if hasattr(font, "path"):
                        font_px = max(self.min_font_px, int(font.size * 0.9))
                        try:
                            font = ImageFont.truetype(font_path, font_px)
                        except Exception:
                            font = ImageFont.load_default()
                        continue
                    else:
                        cur_line = ""  # cannot render; skip
                        break
                last = words.pop()
                if i < len(lines) - 1:
                    next_words = lines[i + 1].split(" ")
                    next_words.insert(0, last)
                    lines[i + 1] = " ".join(next_words)
                else:
                    pass  # drop last word for last line
                cur_line = " ".join(words)
            # Only render if the line fits both horizontally and vertically
            if cur_line:
                w, h = draw.textsize(cur_line, font=font)
                if (w <= max_text_width) and (y + h <= H - margin_px):
                    chosen_rgb = random.choice(self._palette)
                    fill = (chosen_rgb[0], chosen_rgb[1], chosen_rgb[2], 255)
                    draw.text((x, y), cur_line, font=font, fill=fill)
                    rendered_lines.append(cur_line)
                    rendered_colors.append(fill[:3])
                    y += int(h * (1.0 + (self.line_spacing - 1.0)))
                else:
                    break  # never render outside image boundary

        # final small rotation (kept conservative)
        angle = random.uniform(-self.max_rotation, self.max_rotation)
        final = canvas
        if abs(angle) > 0.01:
            rotated = final.rotate(angle, resample=Image.BICUBIC, expand=True)
            canvas_out = Image.new("RGBA", (W, H), (255, 255, 255, 255))
            rx, ry = rotated.size
            paste_x = max(0, (W - rx) // 2)
            paste_y = max(0, (H - ry) // 2)
            canvas_out.paste(rotated, (paste_x, paste_y), rotated)
            final = canvas_out

        image_np = np.array(final)

        label = " ".join([re.sub(r"\s+", " ", ln).strip() for ln in rendered_lines]).strip()
        quality = random.randint(self.quality[0], self.quality[1]) if isinstance(self.quality, (list, tuple)) else int(self.quality)

        roi = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=int)

        data = {
            "image": image_np,
            "label": label,
            "quality": quality,
            "roi": roi,
            "colors": rendered_colors,
        }
        return data


    def init_save(self, root):
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)

    def save(self, root, data, idx):
        image = data["image"]
        label = data["label"]
        quality = data.get("quality", 90)
        colors = data.get("colors", [])

        split_idx = self.split_indexes[idx % len(self.split_indexes)]
        output_dirpath = os.path.join(root, self.splits[split_idx])
        os.makedirs(output_dirpath, exist_ok=True)

        image_filename = f"image_{idx}.jpg"
        image_filepath = os.path.join(output_dirpath, image_filename)
        Image.fromarray(image[..., :3].astype("uint8")).save(image_filepath, quality=quality)

        metadata_filename = "metadata.jsonl"
        metadata_filepath = os.path.join(output_dirpath, metadata_filename)
        metadata = self.format_metadata(image_filename=image_filename, keys=["text_sequence", "text_colors"], values=[label, colors])
        with open(metadata_filepath, "a", encoding="utf-8") as fp:
            json.dump(metadata, fp, ensure_ascii=False)
            fp.write("\n")

    def end_save(self, root):
        return

    def format_metadata(self, image_filename: str, keys: List[str], values: List[Any]):
        assert len(keys) == len(values), "Length does not match: keys({}), values({})".format(len(keys), len(values))
        _gt_parse_v = dict()
        for k, v in zip(keys, values):
            _gt_parse_v[k] = v
        gt_parse = {"gt_parse": _gt_parse_v}
        gt_parse_str = json.dumps(gt_parse, ensure_ascii=False)
        metadata = {"file_name": image_filename, "ground_truth": gt_parse_str}
        return metadata
