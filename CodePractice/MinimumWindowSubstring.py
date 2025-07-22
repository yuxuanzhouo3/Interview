# 给你两个字符串 s 和 t，在 s 中找出包含 t 所有字符的最短子串（包括重复字符），返回这个子串。
import collections
def minWindow(s, t):
    if not s or not t: return ""
    need = collections.Counter(t)
    start = 0
    min_len = float('inf')
    l = r = valid = 0
    window = collections.defaultdict(int)

    while r < len(s):
        c = s[r]
        r += 1
        window[c] += 1
        if window[c] == need[c]:
            valid += 1

        while valid == len(need):
            if r - l < min_len:
                start = l
                min_len = r - l

            d = s[l]
            if window[d] == need[d]:
                valid -= 1
            window[d] -= 1
            l += 1

    return s[start: start+min_len] if min_len!=float('inf') else ""

def test_min_window():
    cases = [
        ("ADOBECODEBANC", "ABC", "BANC"),
        ("a", "a", "a"),
        ("a", "aa", ""),
        ("abdabca", "abc", "abc"),  # ← 修正预期
        ("aaabdabcefaecbef", "abc", "abc"),
        ("", "ABC", ""),
        ("A", "AA", ""),
        ("XYZ", "XYZ", "XYZ"),
        ("", "", ""),
        ("aaabbbccc", "abcabc", "aabbbcc")  # ← 修正预期
    ]

    for i, (s, t, expected) in enumerate(cases):
        result = minWindow(s, t)
        print(f"Test case {i + 1}: {'✅' if result == expected else '❌'} | Output: {result} | Expected: {expected}")


# 调用 test
test_min_window()
