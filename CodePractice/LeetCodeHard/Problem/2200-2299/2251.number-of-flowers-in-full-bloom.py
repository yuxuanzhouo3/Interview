"""
LC 2251. Number of Flowers in Full Bloom — Hard
https://leetcode.com/problems/number-of-flowers-in-full-bloom/
Best: Sort start/end + binary search — O((n+m) log n)
"""

from typing import List
import bisect


class Solution:
    def fullBloomFlowers(self, flowers: List[List[int]], people: List[int]) -> List[int]:
        starts = sorted(a for a, _ in flowers)
        ends = sorted(b for _, b in flowers)
        return [bisect.bisect_right(starts, t) - bisect.bisect_left(ends, t) for t in people]


def _test():
    s = Solution()
    assert s.fullBloomFlowers([[1, 6], [3, 7], [9, 12], [4, 13]], [2, 3, 7, 11]) == [1, 2, 2, 2]
    assert s.fullBloomFlowers([[1, 10], [3, 3]], [3, 3, 2]) == [2, 2, 1]
    print("LC 2251 OK")


if __name__ == "__main__":
    _test()
