"""
LC 2102. Sequentially Ordinal Rank Tracker — Hard
https://leetcode.com/problems/sequentially-ordinal-rank-tracker/
Best: SortedList by (-score, name) — O(log n)
"""

from sortedcontainers import SortedList


class SORTracker:
    def __init__(self):
        self.data = SortedList()
        self.idx = 0

    def add(self, name: str, score: int) -> None:
        self.data.add((-score, name))

    def get(self) -> str:
        ans = self.data[self.idx][1]
        self.idx += 1
        return ans


def _test():
    t = SORTracker()
    t.add("bradford", 2)
    t.add("branford", 3)
    assert t.get() == "branford"
    t.add("alps", 2)
    assert t.get() == "alps"
    t.add("orland", 2)
    assert t.get() == "bradford"
    t.add("orlando", 3)
    assert t.get() == "bradford"
    t.add("alpine", 2)
    assert t.get() == "bradford"
    assert t.get() == "orland"
    print("LC 2102 OK")


if __name__ == "__main__":
    _test()
