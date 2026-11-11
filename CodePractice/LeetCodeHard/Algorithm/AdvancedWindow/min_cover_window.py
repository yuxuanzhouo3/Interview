"""
3. 最小覆盖子串
App: LC 76。
"""

from collections import Counter


def min_window(s, t):
    need = Counter(t)
    missing = len(t)
    l = best_l = 0
    best = float("inf")
    for r, ch in enumerate(s):
        if need[ch] > 0:
            missing -= 1
        need[ch] -= 1
        while missing == 0:
            if r - l + 1 < best:
                best = r - l + 1
                best_l = l
            need[s[l]] += 1
            if need[s[l]] > 0:
                missing += 1
            l += 1
    return "" if best == float("inf") else s[best_l : best_l + best]


def _test():
    assert min_window("ADOBECODEBANC", "ABC") == "BANC"
    assert min_window("a", "a") == "a"
    assert min_window("a", "aa") == ""
    print("min_cover_window OK", min_window("ADOBECODEBANC", "ABC"))


APPLICATION = """
- need/missing：扩到合法再尽量缩；注意 need 可为负表示多余。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
