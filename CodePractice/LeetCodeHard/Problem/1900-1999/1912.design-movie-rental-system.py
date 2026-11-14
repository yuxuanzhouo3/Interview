"""
LC 1912. Design Movie Rental System — Hard
https://leetcode.com/problems/design-movie-rental-system/
Best: ordered sets via SortedList or bisect lists — O(log n)
"""

from typing import List
from collections import defaultdict
import bisect


class _SL:
    """Minimal SortedList substitute."""

    def __init__(self, iterable=None):
        self.a = sorted(iterable) if iterable else []

    def add(self, x):
        bisect.insort(self.a, x)

    def discard(self, x):
        i = bisect.bisect_left(self.a, x)
        if i < len(self.a) and self.a[i] == x:
            self.a.pop(i)

    def __getitem__(self, s):
        return self.a[s]

    def __len__(self):
        return len(self.a)


try:
    from sortedcontainers import SortedList as _SortedList
except Exception:
    _SortedList = _SL


class MovieRentingSystem:
    def __init__(self, n: int, entries: List[List[int]]):
        self.price = {}
        self.unrented = defaultdict(_SortedList)
        self.rented = _SortedList()
        for shop, movie, price in entries:
            self.price[(shop, movie)] = price
            self.unrented[movie].add((price, shop))

    def search(self, movie: int) -> List[int]:
        return [shop for _, shop in list(self.unrented[movie][:5])]

    def rent(self, shop: int, movie: int) -> None:
        p = self.price[(shop, movie)]
        self.unrented[movie].discard((p, shop))
        self.rented.add((p, shop, movie))

    def drop(self, shop: int, movie: int) -> None:
        p = self.price[(shop, movie)]
        self.rented.discard((p, shop, movie))
        self.unrented[movie].add((p, shop))

    def report(self) -> List[List[int]]:
        return [[shop, movie] for _, shop, movie in list(self.rented[:5])]


def _test():
    sys = MovieRentingSystem(3, [[0, 1, 5], [0, 2, 6], [0, 3, 7], [1, 1, 4], [1, 2, 7], [2, 1, 5]])
    assert sys.search(1) == [1, 0, 2]
    sys.rent(0, 1)
    sys.rent(1, 2)
    assert sys.report() == [[0, 1], [1, 2]]
    sys.drop(1, 2)
    assert sys.search(2) == [0, 1]
    print("LC 1912 OK")


if __name__ == "__main__":
    _test()
