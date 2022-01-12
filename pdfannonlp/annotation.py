from enum import Enum
from .rect import Rect
from .text import Char


class AnnotType(Enum):
    RECT = 0,
    TEXT = 1,


class Annotation:
    def __init__(self, type: AnnotType, label: str, rect: Rect):
        self.type = type
        self.label: str = label
        self.rect: Rect = rect

    def extract_text(self, chars: list[Char]) -> list[Char]:
        """
        Extract text from annotation
        """
        def find_index(f, _chars: list[Char], start: int):
            for i in range(start, len(_chars)):
                if f(_chars[i]):
                    return i
            return -1

        if self.type == AnnotType.RECT:
            return [self.rect.contains(c.bbox.center()) for c in chars]
        elif self.type == AnnotType.TEXT:
            rect = self.rect
            c1 = Rect(rect.x1, rect.y1, rect.x1, rect.y1)
            c2 = Rect(rect.x2, rect.y2, rect.x2, rect.y2)
            index1 = find_index(lambda c: c.bbox.contains(c1), chars, 0)
            assert index1 >= 0
            index2 = find_index(lambda c: c.bbox.contains(c2), chars, index1)
            assert index2 >= 0
            return chars[index1:index2 + 1]


    @staticmethod
    def from_json(json_data):
        label = json_data["label"]
        if "bbox" in json_data:
            type: AnnotType = AnnotType.RECT
            rect = Rect(*json_data["bbox"])
        elif "char_range" in json_data:
            type: AnnotType = AnnotType.TEXT
            rect = Rect(*json_data["char_range"])
        else:
            raise Exception("Invalid annotation")
        return Annotation(type, label, rect)
