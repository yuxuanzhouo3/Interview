"""
4. 替换容错最长重复字符
App: LC 424 / 1004（1 的最长翻转）。
"""

from collections import defaultdict


def character_replacement(s, k):
    cnt = defaultdict(int)
    l = ans = maxf = 0
    for r, ch in enumerate(s):
        cnt[ch] += 1
        maxf = max(maxf, cnt[ch])
        while (r - l + 1) - maxf > k:
            cnt[s[l]] -= 1
            l += 1
        ans = max(ans, r - l + 1)
    return ans


def longest_ones(nums, k):
    """最多翻 k 个 0。"""
    l = zeros = ans = 0
    for r, x in enumerate(nums):
        zeros += x == 0
        while zeros > k:
            zeros -= nums[l] == 0
            l += 1
        ans = max(ans, r - l + 1)
    return ans


def _test():
    assert character_replacement("AABABBA", 1) == 4
    assert longest_ones([1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0], 2) == 6
    print("char_replace_window OK")


APPLICATION = """
- 条件：(窗长 - 窗内最多字符频次) <= k。
- maxf 可不减（历史最大值仍正确，因 ans 只增）。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
