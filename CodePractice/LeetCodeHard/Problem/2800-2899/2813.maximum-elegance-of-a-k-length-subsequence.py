"""
LC 2813. Maximum Elegance of a K-Length Subsequence — Hard
https://leetcode.com/problems/maximum-elegance-of-a-k-length-subsequence/
Best: Sort by profit; greedy replace duplicate categories — O(n log n)
"""

from typing import List


class Solution:
    def findMaximumElegance(self, items: List[List[int]], k: int) -> int:
        items.sort(key=lambda x: -x[0])
        seen = set()
        dup = []  # profits of duplicate-category picks (candidates to drop)
        total = 0
        ans = 0
        for i, (profit, cat) in enumerate(items):
            if i < k:
                total += profit
                if cat in seen:
                    dup.append(profit)
                else:
                    seen.add(cat)
                ans = total + len(seen) * len(seen)
            else:
                if cat not in seen and dup:
                    total += profit - dup.pop()
                    seen.add(cat)
                    ans = max(ans, total + len(seen) * len(seen))
        return ans


def _test():
    s = Solution()
    assert s.findMaximumElegance([[3, 2], [5, 1], [10, 1]], 2) == 17
    assert s.findMaximumElegance([[3, 1], [3, 1], [2, 2], [5, 3]], 3) == 19
    assert s.findMaximumElegance([[1, 1], [2, 1], [3, 1]], 3) == 7
    print("LC 2813 OK")


if __name__ == "__main__":
    _test()
