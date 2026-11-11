"""
8. 滑动窗口最大值
App: LC 239。
"""

from collections import deque


def max_sliding_window(nums, k):
    dq = deque()  # 递减下标
    out = []
    for i, x in enumerate(nums):
        while dq and nums[dq[-1]] <= x:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:
            dq.popleft()
        if i >= k - 1:
            out.append(nums[dq[0]])
    return out


def _test():
    assert max_sliding_window([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]
    assert max_sliding_window([1], 1) == [1]
    print("sliding_window_max OK")


APPLICATION = """
- 队头是窗内最大；出界弹头，不单调弹尾。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
