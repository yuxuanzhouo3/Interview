"""
5. 固定长度窗
App: LC 1456 / 567 排列包含。
"""

from collections import Counter


def max_vowels(s, k):
    vowels = set("aeiou")
    cur = sum(ch in vowels for ch in s[:k])
    ans = cur
    for i in range(k, len(s)):
        cur += (s[i] in vowels) - (s[i - k] in vowels)
        ans = max(ans, cur)
    return ans


def check_inclusion(s1, s2):
    """s2 是否含 s1 的排列。"""
    m, n = len(s1), len(s2)
    if m > n:
        return False
    need = Counter(s1)
    win = Counter(s2[:m])
    if win == need:
        return True
    for i in range(m, n):
        win[s2[i]] += 1
        win[s2[i - m]] -= 1
        if win[s2[i - m]] == 0:
            del win[s2[i - m]]
        if win == need:
            return True
    return False


def _test():
    assert max_vowels("abciiidef", 3) == 3
    assert check_inclusion("ab", "eidbaooo") is True
    assert check_inclusion("ab", "eidboaoo") is False
    print("fixed_window OK")


APPLICATION = """
- 进一出一维护统计；排列题用 Counter/差分数组判等。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
