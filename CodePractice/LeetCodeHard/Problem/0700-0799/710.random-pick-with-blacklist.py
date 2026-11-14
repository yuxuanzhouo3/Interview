"""
LC 710. Random Pick with Blacklist — Hard
Best: Remap blacklist into [0, n-m) — O(B) init, O(1) pick
"""

import random
from typing import List


class Solution:
    def __init__(self, n: int, blacklist: List[int]):
        self.bound = n - len(blacklist)
        black = set(blacklist)
        self.map = {}
        whites = iter(x for x in range(self.bound, n) if x not in black)
        for b in blacklist:
            if b < self.bound:
                self.map[b] = next(whites)

    def pick(self) -> int:
        x = random.randrange(self.bound)
        return self.map.get(x, x)


def _test():
    s = Solution(7, [2, 3, 5])
    for _ in range(20):
        assert s.pick() in {0, 1, 4, 6}
    print("LC 710 OK")


if __name__ == "__main__":
    _test()
