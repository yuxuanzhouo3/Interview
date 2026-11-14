"""
LC 683. K Empty Slots — Hard
https://leetcode.com/problems/k-empty-slots/
Best: day[] + sliding window on positions — O(n)
"""

from typing import List


class Solution:
    def kEmptySlots(self, bulbs: List[int], k: int) -> int:
        n = len(bulbs)
        day = [0] * n
        for d, pos in enumerate(bulbs, 1):
            day[pos - 1] = d

        ans = float("inf")
        left = 0
        right = k + 1
        i = 1
        while right < n:
            if day[i] < day[left] or day[i] <= day[right]:
                if i == right:
                    ans = min(ans, max(day[left], day[right]))
                left, right = i, i + k + 1
                i = left + 1
                continue
            i += 1
        return ans if ans != float("inf") else -1


def _test():
    s = Solution()
    assert s.kEmptySlots([1, 3, 2], 1) == 2
    assert s.kEmptySlots([1, 2, 3], 1) == -1
    print("LC 683 OK")


if __name__ == "__main__":
    _test()
