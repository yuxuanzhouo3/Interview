"""
LC 2551. Put Marbles in Bags — Hard
https://leetcode.com/problems/put-marbles-in-bags/
Best: Sort adjacent pair sums; take k-1 largest/smallest — O(n log n)
"""

from typing import List


class Solution:
    def putMarbles(self, weights: List[int], k: int) -> int:
        if k == 1:
            return 0
        pairs = [weights[i] + weights[i + 1] for i in range(len(weights) - 1)]
        pairs.sort()
        return sum(pairs[-(k - 1):]) - sum(pairs[: k - 1])


def _test():
    s = Solution()
    assert s.putMarbles([1, 3, 5, 1], 2) == 4
    assert s.putMarbles([1, 3], 2) == 0
    print("LC 2551 OK")


if __name__ == "__main__":
    _test()
