"""
1. 最长无重复子串
App: LC 3。
"""


def length_of_longest_substring(s):
    last, l, ans = {}, 0, 0
    for r, ch in enumerate(s):
        if ch in last and last[ch] >= l:
            l = last[ch] + 1
        last[ch] = r
        ans = max(ans, r - l + 1)
    return ans


def _test():
    assert length_of_longest_substring("abcabcbb") == 3
    assert length_of_longest_substring("bbbbb") == 1
    assert length_of_longest_substring("pwwkew") == 3
    print("no_repeat_window OK", length_of_longest_substring("abcabcbb"))


APPLICATION = """
- 右扩；若字符上次出现在窗内则左指针跳到上次+1。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
