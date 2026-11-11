"""
7. 前缀和 + 单调队列
App: LC 862 最短子数组和 ≥K；209 思想。
"""

from collections import deque


def shortest_subarray(nums, k):
    n = len(nums)
    pref = [0] * (n + 1)
    for i, x in enumerate(nums):
        pref[i + 1] = pref[i] + x
    dq = deque()  # 存下标，pref 递增
    ans = n + 1
    for i, p in enumerate(pref):
        while dq and p - pref[dq[0]] >= k:
            ans = min(ans, i - dq.popleft())
        while dq and pref[dq[-1]] >= p:
            dq.pop()
        dq.append(i)
    return ans if ans <= n else -1


def min_subarray_len(target, nums):
    """正数数组：经典双指针。"""
    l = s = 0
    ans = float("inf")
    for r, x in enumerate(nums):
        s += x
        while s >= target:
            ans = min(ans, r - l + 1)
            s -= nums[l]
            l += 1
    return 0 if ans == float("inf") else ans


def _test():
    assert shortest_subarray([2, -1, 2], 3) == 3
    assert shortest_subarray([1, 2], 4) == -1
    assert min_subarray_len(7, [2, 3, 1, 2, 4, 3]) == 2
    print("prefix_mono_deque OK")


APPLICATION = """
- 含负数不能裸双指针 → 前缀 + 递增单调队列。
- 队头满足和 ≥K 则弹出更新最短。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
