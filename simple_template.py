"""
Custom SynthDoG-style Generator
"""

import json
import os
import re
import numpy as np
from typing import Any, List

from elements import Background, Document
from PIL import Image
from synthtiger import components, layers, templates


class SynthGenerator(templates.Template):
    def __init__(self, config=None, split_ratio: List[float] = []):
        super().__init__(config)
        if config is None:
            config = {}

        # config
        self.quality = config.get("quality", [70, 95])
        self.landscape = config.get("landscape", 0.5)
        self.short_size = config.get("short_size", [224, 350])  # 👈 clamp here
        self.aspect_ratio = config.get("aspect_ratio", [1.0, 1.5])

        # background and document modules
        self.background = Background(config.get("background", {}))
        self.document = Document(config.get("document", {}))

        # visual effects
        self.effect = components.Iterator(
            [
                components.Switch(components.RGB()),
                components.Switch(components.Shadow()),
                components.Switch(components.Contrast()),
                components.Switch(components.Brightness()),
                components.Switch(components.MotionBlur()),
                components.Switch(components.GaussianBlur()),
            ],
            **config.get("effect", {}),
        )

        # dataset splits
        self.splits = ["train", "validation", "test"]
        self.split_ratio = split_ratio if split_ratio else [0.9, 0.05, 0.05]
        self.split_indexes = np.random.choice(3, size=10000, p=self.split_ratio)

    def generate(self):
        # choose orientation + aspect
        landscape = np.random.rand() < self.landscape
        short_size = np.random.randint(self.short_size[0], self.short_size[1] + 1)
        aspect_ratio = np.random.uniform(self.aspect_ratio[0], self.aspect_ratio[1])
        long_size = int(short_size * aspect_ratio)
        size = (long_size, short_size) if landscape else (short_size, long_size)

        # background and text layers
        bg_layer = self.background.generate(size)
        paper_layer, text_layers, texts = self.document.generate(size)

        # group text + paper, then overlay onto background
        document_group = layers.Group([*text_layers, paper_layer])
        document_space = np.clip(size - document_group.size, 0, None)
        document_group.left = np.random.randint(document_space[0] + 1)
        document_group.top = np.random.randint(document_space[1] + 1)
        roi = np.array(paper_layer.quad, dtype=int)

        layer = layers.Group([*document_group.layers, bg_layer]).merge()
        self.effect.apply([layer])

        # final render
        image = layer.output(bbox=[0, 0, *size])
        label = " ".join(texts).strip()
        label = re.sub(r"\s+", " ", label)  # normalize whitespace
        quality = np.random.randint(self.quality[0], self.quality[1] + 1)

        return {
            "image": image,
            "label": label,
            "quality": quality,
            "roi": roi,
        }

    def init_save(self, root):
        os.makedirs(root, exist_ok=True)

    def save(self, root, data, idx):
        image = data["image"]
        label = data["label"]
        quality = data["quality"]
        roi = data["roi"]

        # split handling
        split_idx = self.split_indexes[idx % len(self.split_indexes)]
        output_dirpath = os.path.join(root, self.splits[split_idx])

        # save image
        image_filename = f"image_{idx}.jpg"
        image_filepath = os.path.join(output_dirpath, image_filename)
        os.makedirs(os.path.dirname(image_filepath), exist_ok=True)
        Image.fromarray(image[..., :3].astype(np.uint8)).save(image_filepath, quality=quality)

        # save metadata
        metadata_filename = "metadata.jsonl"
        metadata_filepath = os.path.join(output_dirpath, metadata_filename)
        os.makedirs(os.path.dirname(metadata_filepath), exist_ok=True)

        metadata = self.format_metadata(
            image_filename=image_filename,
            keys=["text_sequence"],
            values=[label]
        )
        with open(metadata_filepath, "a", encoding="utf-8") as fp:
            json.dump(metadata, fp, ensure_ascii=False)
            fp.write("\n")

    def end_save(self, root):
        pass

    def format_metadata(self, image_filename: str, keys: List[str], values: List[Any]):
        assert len(keys) == len(values)
        gt_parse = {"gt_parse": dict(zip(keys, values))}
        gt_parse_str = json.dumps(gt_parse, ensure_ascii=False)
        return {"file_name": image_filename, "ground_truth": gt_parse_str}
