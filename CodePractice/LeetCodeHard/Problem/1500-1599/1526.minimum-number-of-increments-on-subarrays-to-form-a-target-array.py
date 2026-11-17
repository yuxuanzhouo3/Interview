"""
LC 1526. Minimum Number of Increments on Subarrays to Form a Target Array — Hard
https://leetcode.com/problems/minimum-number-of-increments-on-subarrays-to-form-a-target-array/
Best: Greedy diff — O(n)
"""

from typing import List


class Solution:
    def minNumberOperations(self, target: List[int]) -> int:
        ans = target[0]
        for i in range(1, len(target)):
            if target[i] > target[i - 1]:
                ans += target[i] - target[i - 1]
        return ans


def _test():
    s = Solution()
    assert s.minNumberOperations([1, 2, 3, 2, 1]) == 3
    assert s.minNumberOperations([3, 1, 1, 2]) == 4
    assert s.minNumberOperations([3, 1, 5, 4, 2]) == 7
    print("LC 1526 OK")


if __name__ == "__main__":
    _test()
