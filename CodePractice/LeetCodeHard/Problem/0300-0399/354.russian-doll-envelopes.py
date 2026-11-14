"""
LC 354. Russian Doll Envelopes — Hard
Best: Sort w asc, h desc; LIS on h — O(n log n)
"""

from typing import List
import bisect


class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        envelopes.sort(key=lambda x: (x[0], -x[1]))
        tails = []
        for _, h in envelopes:
            i = bisect.bisect_left(tails, h)
            if i == len(tails):
                tails.append(h)
            else:
                tails[i] = h
        return len(tails)


def _test():
    s = Solution()
    assert s.maxEnvelopes([[5, 4], [6, 4], [6, 7], [2, 3]]) == 3
    assert s.maxEnvelopes([[1, 1], [1, 1], [1, 1]]) == 1
    print("LC 354 OK")


if __name__ == "__main__":
    _test()
