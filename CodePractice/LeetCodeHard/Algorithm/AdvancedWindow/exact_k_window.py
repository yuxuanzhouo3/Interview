"""
6. 恰好 K 种 —— 双窗差分
App: LC 992。
"""

from collections import defaultdict


def at_most(nums, k):
    cnt = defaultdict(int)
    l = ans = 0
    for r, x in enumerate(nums):
        cnt[x] += 1
        while len(cnt) > k:
            cnt[nums[l]] -= 1
            if cnt[nums[l]] == 0:
                del cnt[nums[l]]
            l += 1
        ans += r - l + 1  # 以 r 结尾的合法子数组个数
    return ans


def subarrays_with_k_distinct(nums, k):
    return at_most(nums, k) - at_most(nums, k - 1)


def _test():
    assert subarrays_with_k_distinct([1, 2, 1, 2, 3], 2) == 7
    assert subarrays_with_k_distinct([1, 2, 1, 3, 4], 3) == 3
    print("exact_k_window OK", subarrays_with_k_distinct([1, 2, 1, 2, 3], 2))


APPLICATION = """
- 「恰好 K」=「至多 K」−「至多 K-1」。
- atMost 返回子数组个数时用 ans += r-l+1。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
