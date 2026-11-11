"""
9. 窗内 max-min ≤ limit 的最长连续子数组
App: LC 1438。
"""

from collections import deque


def longest_subarray(nums, limit):
    maxq, minq = deque(), deque()
    l = ans = 0
    for r, x in enumerate(nums):
        while maxq and maxq[-1] < x:
            maxq.pop()
        while minq and minq[-1] > x:
            minq.pop()
        maxq.append(x)
        minq.append(x)
        while maxq[0] - minq[0] > limit:
            if maxq[0] == nums[l]:
                maxq.popleft()
            if minq[0] == nums[l]:
                minq.popleft()
            l += 1
        ans = max(ans, r - l + 1)
    return ans


def _test():
    assert longest_subarray([8, 2, 4, 7], 4) == 2
    assert longest_subarray([10, 1, 2, 4, 7, 2], 5) == 4
    print("abs_diff_limit OK", longest_subarray([8, 2, 4, 7], 4))


APPLICATION = """
- 双单调队列同时维护窗 max/min；极差超限左缩。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
