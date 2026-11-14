"""
LC 768. Max Chunks To Make Sorted II — Hard
Best: Max-left vs min-right — O(n)
"""

from typing import List


class Solution:
    def maxChunksToSorted(self, arr: List[int]) -> int:
        n = len(arr)
        max_l = [0] * n
        min_r = [0] * n
        max_l[0] = arr[0]
        for i in range(1, n):
            max_l[i] = max(max_l[i - 1], arr[i])
        min_r[-1] = arr[-1]
        for i in range(n - 2, -1, -1):
            min_r[i] = min(min_r[i + 1], arr[i])
        ans = 1
        for i in range(n - 1):
            if max_l[i] <= min_r[i + 1]:
                ans += 1
        return ans


def _test():
    s = Solution()
    assert s.maxChunksToSorted([5, 4, 3, 2, 1]) == 1
    assert s.maxChunksToSorted([2, 1, 3, 4, 4]) == 4
    print("LC 768 OK")


if __name__ == "__main__":
    _test()
