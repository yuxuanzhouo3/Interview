"""
LC 381. Insert Delete GetRandom O(1) - Duplicates allowed — Hard
Best: val→set of indices + list — O(1) avg
"""

import random
from collections import defaultdict


class RandomizedCollection:
    def __init__(self):
        self.vals = []
        self.idx = defaultdict(set)

    def insert(self, val: int) -> bool:
        self.idx[val].add(len(self.vals))
        self.vals.append(val)
        return len(self.idx[val]) == 1

    def remove(self, val: int) -> bool:
        if not self.idx[val]:
            return False
        i = self.idx[val].pop()
        last = self.vals[-1]
        self.vals[i] = last
        self.idx[last].add(i)
        self.idx[last].discard(len(self.vals) - 1)
        self.vals.pop()
        return True

    def getRandom(self) -> int:
        return random.choice(self.vals)


def _test():
    r = RandomizedCollection()
    assert r.insert(1) is True
    assert r.insert(1) is False
    assert r.insert(2) is True
    x = r.getRandom()
    assert x in (1, 2)
    assert r.remove(1) is True
    print("LC 381 OK")


if __name__ == "__main__":
    _test()
