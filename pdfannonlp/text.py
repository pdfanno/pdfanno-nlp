from enum import Enum

from .rect import Rect


class CharScript(Enum):  # superscript, subscript
    NONE = 0,
    SUPER = 1,
    SUB = 2,


class Char:
    def __init__(self, value: str, bbox: Rect, font_size: int, script=CharScript.NONE):
        self.value: str = value
        self.bbox: Rect = bbox
        self.font_size: int = font_size
        self.script: CharScript = script

    def __str__(self):
        return f"{self.value}"

    def html_text(self):
        if self.script == CharScript.SUB:
            return f"<sub>{self.value}</sub>"
        elif self.script == CharScript.SUPER:
            return f"<sup>{self.value}</sup>"
        else:
            return self.value


class Word:
    def __init__(self, bbox: Rect, chars: list[Char]):
        self.bbox = bbox
        self.chars = chars
        self.tag = None

    def __getitem__(self, i):
        return self.chars[i]

    def __str__(self):
        return self.value()

    def __repr__(self):
        return self.value()

    def text(self):
        return "".join(c.value for c in self.chars)

    def set_scripts(self):
        """
        Set CharScript for each char
        """
        # TODO: vertical case
        chars = self.chars
        base = chars[0]
        for c in chars:
            if c.font_size > base.font_size:
                base = c
        for c in chars:
            if c.font_size < base.font_size * 0.8:
                if c.bbox.center().y1 < base.bbox.center().y1:
                    c.script = CharScript.SUPER
                else:
                    c.script = CharScript.SUB
            else:
                c.script = CharScript.NONE

    @staticmethod
    def from_chars(chars: list[Char]):
        bbox = chars[0].bbox
        for c in chars:
            bbox = bbox.union(c.bbox)
        return Word(bbox, chars)

    @staticmethod
    def tokenize(chars: list[Char], threshold=0.3):
        """
        Convert a sequence of chars to that of words
        """
        def newline(box1, box2):
            h1 = box1.height
            h2 = box2.height
            return box1.overlap_y(box2) < 0.3 * min(h1, h2) or box1.x1 > box2.x2

        words = []
        buffer = []
        bbox = None
        for i, c in enumerate(chars):
            buffer.append(c)
            if bbox is None:
                bbox = c.bbox
            else:
                bbox = bbox.union(c.bbox)
            if i + 1 < len(chars):
                mean_w = bbox.width / len(buffer)
                # TODO: newline check is necessary for word detection?
                is_token = c.bbox.x2 + mean_w * threshold < chars[i + 1].bbox.x1 or newline(bbox, chars[i + 1].bbox)
            else:
                is_token = True
            if is_token:
                words.append(Word.from_chars(buffer))
                buffer = []
                bbox = None
        return words


class Text:
    def __init__(self, bbox: Rect, words: list[Word]):
        self.bbox = bbox
        self.words = words

    def __str__(self):
        return self.value()

    def __repr__(self):
        return self.value()

    def value(self, delim=" "):
        return delim.join(w.value() for w in self.words)

    @staticmethod
    def from_words(words: list[Word]):
        bbox = words[0].bbox
        for w in words:
            bbox = bbox.union(w.bbox)
        return Text(bbox, words)

    @staticmethod
    def tokenize(words: list[Word]):
        """
        Convert a sequence of words to that of texts
        """
        texts = []
        buffer = []
        for i, w in enumerate(words):
            buffer.append(w)
            if i + 1 < len(words):
                box1 = w.bbox
                h1 = box1.height
                box2 = words[i + 1].bbox
                h2 = box2.height
                is_token = box1.overlap_y(box2) < 0.3 * min(h1, h2) or box1.x1 > box2.x2
            else:
                is_token = True
            if is_token:
                texts.append(Text.from_words(buffer))
                buffer = []
        return texts
