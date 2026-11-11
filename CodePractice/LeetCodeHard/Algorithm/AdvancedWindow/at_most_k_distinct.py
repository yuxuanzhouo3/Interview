"""
2. 至多 K 种字符的最长子串
App: LC 340 / 159。
"""

from collections import defaultdict


def longest_at_most_k(s, k):
    if k <= 0:
        return 0
    cnt = defaultdict(int)
    l = ans = 0
    for r, ch in enumerate(s):
        cnt[ch] += 1
        while len(cnt) > k:
            cnt[s[l]] -= 1
            if cnt[s[l]] == 0:
                del cnt[s[l]]
            l += 1
        ans = max(ans, r - l + 1)
    return ans


def _test():
    assert longest_at_most_k("eceba", 2) == 3
    assert longest_at_most_k("aa", 1) == 2
    print("at_most_k_distinct OK", longest_at_most_k("eceba", 2))


APPLICATION = """
- 经典可变窗：违规 while 左缩；恰好 K = atMost(K)-atMost(K-1)。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
