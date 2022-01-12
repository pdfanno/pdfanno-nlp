import warnings
from pathlib import Path
import json
import gzip
import re
import unicodedata

from .rect import Rect
from .text import Char, Word, Text
from .annotation import Annotation


class PDF:

    def __init__(self, filename: str):
        self.filename = filename
        self.pages: list[Page] = []

    def load_annotation(self, filename: str):
        with open(filename, encoding="utf-8") as f:
            json_data = json.load(f)
        assert len(self.pages) == len(json_data)

        for i, annots in enumerate(json_data):
            page = self.pages[i]
            for ann_json in annots:
                ann = Annotation.from_json(ann_json)
                page.annots.append(ann)

    @staticmethod
    def load_text(filename: str):
        filename = str(Path(filename).absolute())
        with gzip.open(filename, 'rt', encoding='utf-8') as f:
            data_dict = json.load(f)

        pdf = PDF(filename)
        for i, page_dict in enumerate(data_dict["pages"]):
            width = page_dict["width"]
            height = page_dict["height"]
            page = Page(width, height)
            pdf.pages.append(page)

            text_dict = page_dict["text"]
            values = text_dict["values"]
            x1s = text_dict["x1s"]
            y1s = text_dict["y1s"]
            x2s = text_dict["x2s"]
            y2s = text_dict["y2s"]

            chars = []
            for k in range(len(values)):
                bbox = Rect(x1s[k], y1s[k], x2s[k], y2s[k])
                bbox = bbox.scale(1 / width, 1 / height)
                char = Char(values[k], bbox)
                chars.append(char)
            page.chars = chars
            page.words = Word.tokenize(chars)
        return pdf


class Page:

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.chars: list[Char] = []
        self.words: list[Word] = []
        self.annots: list[Annotation] = []
