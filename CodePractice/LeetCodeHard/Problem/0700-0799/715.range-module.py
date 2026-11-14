"""
LC 715. Range Module — Hard
Best: Sorted intervals merge / SortedDict — O(n) simple
"""

import bisect


class RangeModule:
    def __init__(self):
        self.intervals = []  # sorted [l,r)

    def addRange(self, left: int, right: int) -> None:
        new = []
        placed = False
        for a, b in self.intervals:
            if b < left:
                new.append([a, b])
            elif right < a:
                if not placed:
                    new.append([left, right]); placed = True
                new.append([a, b])
            else:
                left = min(left, a)
                right = max(right, b)
        if not placed:
            new.append([left, right])
        self.intervals = new

    def queryRange(self, left: int, right: int) -> bool:
        for a, b in self.intervals:
            if a <= left and right <= b:
                return True
            if a >= right:
                break
        return False

    def removeRange(self, left: int, right: int) -> None:
        new = []
        for a, b in self.intervals:
            if b <= left or a >= right:
                new.append([a, b])
            else:
                if a < left:
                    new.append([a, left])
                if b > right:
                    new.append([right, b])
        self.intervals = new


def _test():
    r = RangeModule()
    r.addRange(10, 20)
    r.removeRange(14, 16)
    assert r.queryRange(10, 14) is True
    assert r.queryRange(13, 15) is False
    assert r.queryRange(16, 17) is True
    print("LC 715 OK")


if __name__ == "__main__":
    _test()
