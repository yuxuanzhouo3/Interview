"""
LC 1782. Count Pairs Of Nodes — Hard
https://leetcode.com/problems/count-pairs-of-nodes/
Best: Degree sort + edge frequency correction — O(n log n + q n + m)
"""

from collections import Counter
from typing import List
import bisect


class Solution:
    def countPairs(
        self, n: int, edges: List[List[int]], queries: List[int]
    ) -> List[int]:
        deg = [0] * (n + 1)
        cnt = Counter()
        for a, b in edges:
            if a > b:
                a, b = b, a
            deg[a] += 1
            deg[b] += 1
            cnt[(a, b)] += 1
        sorted_deg = sorted(deg[1:])
        ans = []
        for q in queries:
            # count pairs i<j with deg[i]+deg[j] > q, then subtract false positives
            total = 0
            l, r = 0, n - 1
            while l < r:
                if sorted_deg[l] + sorted_deg[r] > q:
                    total += r - l
                    r -= 1
                else:
                    l += 1
            for (a, b), c in cnt.items():
                if deg[a] + deg[b] > q and deg[a] + deg[b] - c <= q:
                    total -= 1
            ans.append(total)
        return ans


def _test():
    s = Solution()
    assert s.countPairs(4, [[1, 2], [2, 4], [1, 3], [2, 3], [2, 1]], [2, 3]) == [6, 5]
    assert s.countPairs(5, [[1, 5], [1, 5], [3, 4], [2, 5], [1, 3], [5, 1], [2, 3], [2, 5]], [1, 2, 3, 4, 5]) == [
        10, 10, 9, 8, 6
    ]
    print("LC 1782 OK")


if __name__ == "__main__":
    _test()
