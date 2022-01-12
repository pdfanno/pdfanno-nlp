class Rect:

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def __str__(self):
        return f"[{self.x1} {self.y1} {self.x2} {self.y2}]"

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def area(self):
        return self.width * self.height

    def int(self):
        return Rect(int(self.x1), int(self.y1), int(self.x2), int(self.y2))

    def center(self):
        x = (self.x1 + self.x2) / 2
        y = (self.y1 + self.y2) / 2
        return Rect(x, y, x, y)

    def clip(self, min_value, max_value):
        x1 = min(max_value, max(self.x1, min_value))
        y1 = min(max_value, max(self.y1, min_value))
        x2 = min(max_value, max(self.x2, min_value))
        y2 = min(max_value, max(self.y2, min_value))
        return Rect(x1, y1, x2, y2)

    def union(self, o):
        x1 = min(self.x1, o.x1)
        y1 = min(self.y1, o.y1)
        x2 = max(self.x2, o.x2)
        y2 = max(self.y2, o.y2)
        return Rect(x1, y1, x2, y2)

    def contains(self, o):
        return self.y1 <= o.y1 and o.y2 <= self.y2 and self.x1 <= o.x1 and o.x2 <= self.x2

    def move(self, w, h):
        return Rect(self.x1 + w, self.y1 + h, self.x2 + w, self.y2 + h)

    def scale(self, scale_w, scale_h):
        return Rect(self.x1 * scale_w, self.y1 * scale_h, self.x2 * scale_w, self.y2 * scale_h)

    def list(self):
        return [self.x1, self.y1, self.x2, self.y2]

    def tuple(self):
        return self.x1, self.y1, self.x2, self.y2

    def overlap_x(self, o):
        return min(self.x2, o.x2) - max(self.x1, o.x1)

    def overlap_y(self, o):
        return min(self.y2, o.y2) - max(self.y1, o.y1)

    def intersection(self):
        # https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
        x1, y1, x2, y2 = self.tuple()
        assert x1 <= x2 and y1 <= y2
        x3, y3, x4, y4 = x1, y2, x2, y1
        a = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
        b = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        x = a / b
        a = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
        y = a / b
        return x, y

    @staticmethod
    def from_rects(rects: list):
        r = rects[0]
        for o in rects:
            r = r.union(o)
        return r
