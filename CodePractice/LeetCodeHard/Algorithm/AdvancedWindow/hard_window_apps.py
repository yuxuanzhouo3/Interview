"""
10. Hard 滑窗综合：串联词 / 平衡字符窗
App: LC 30 / 1234 思想。
"""

from collections import Counter


def find_substring(s, words):
    """LC 30: 串联所有词的起始下标（词长相同）。"""
    if not s or not words:
        return []
    wlen = len(words[0])
    total = wlen * len(words)
    need = Counter(words)
    ans = []
    for offset in range(wlen):
        l = offset
        win = Counter()
        count = 0
        for r in range(offset, len(s) - wlen + 1, wlen):
            w = s[r : r + wlen]
            if w in need:
                win[w] += 1
                count += 1
                while win[w] > need[w]:
                    left_w = s[l : l + wlen]
                    win[left_w] -= 1
                    l += wlen
                    count -= 1
                if count == len(words):
                    ans.append(l)
            else:
                win.clear()
                count = 0
                l = r + wlen
    return ans


def balanced_string(s):
    """LC 1234: 替换最短子串使 QWER 各 n/4。"""
    n = len(s)
    need = n // 4
    cnt = Counter(s)
    if all(cnt[c] <= need for c in "QWER"):
        return 0
    l = 0
    ans = n
    for r, ch in enumerate(s):
        cnt[ch] -= 1
        while all(cnt[c] <= need for c in "QWER"):
            ans = min(ans, r - l + 1)
            cnt[s[l]] += 1
            l += 1
    return ans


def _test():
    assert find_substring("barfoothefoobarman", ["foo", "bar"]) == [0, 9]
    assert balanced_string("QWER") == 0
    assert balanced_string("QQWE") == 1
    print("hard_window_apps OK")


APPLICATION = """
- 30: 按词长分 offset，窗以「词」为单位滑动。
- 1234: 窗外已合法 → 最小化窗长（替换这段）。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
