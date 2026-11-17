"""
LC 3948. Lexicographically Maximum MEX Array — Hard
https://leetcode.com/problems/lexicographically-maximum-mex-array/
Best: greedy cut at first position that realizes maximal prefix MEX
"""
from collections import defaultdict, deque


class Solution:
    def maximumMEX(self, nums: list[int]) -> list[int]:
        at: dict[int, deque[int]] = defaultdict(deque)
        for i, x in enumerate(nums):
            at[x].append(i)
        res = []
        pos = 0
        while 0 in at:
            val = 0
            now = pos
            while val in at:
                now = max(now, at[val][0])
                val += 1
            res.append(val)
            while pos <= now:
                x = nums[pos]
                at[x].popleft()
                if not at[x]:
                    del at[x]
                pos += 1
        while pos < len(nums):
            res.append(0)
            pos += 1
        return res


def _test():
    s = Solution()
    assert s.maximumMEX([0, 1, 0]) == [2, 1]
    assert s.maximumMEX([1, 0, 2]) == [3]
    assert s.maximumMEX([3, 1]) == [0, 0]
    print("LC 3948 OK")


if __name__ == "__main__":
    _test()
