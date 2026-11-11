"""
4. Manacher (最长回文子串 O(n))
Template: query(longest / radius) / update(rebuild) / add(string) / del
App: 回文判定加速；LC 5 Longest Palindromic Substring。
"""


class Manacher:
    def __init__(self, s=""):
        self.raw = s
        self.t = ""
        self.p = []
        if s:
            self.add(s)

    def _prep(self, s):
        return "^#" + "#".join(s) + "#$"

    def add(self, s):
        self.raw = s
        t = self._prep(s)
        self.t = t
        n = len(t)
        p = [0] * n
        center = right = 0
        for i in range(1, n - 1):
            mirror = 2 * center - i
            if i < right:
                p[i] = min(right - i, p[mirror])
            while t[i + 1 + p[i]] == t[i - 1 - p[i]]:
                p[i] += 1
            if i + p[i] > right:
                center, right = i, i + p[i]
        self.p = p
        return p

    def query(self, mode="longest"):
        if not self.p:
            return "" if mode == "longest" else []
        if mode == "radius":
            return list(self.p)
        i = max(range(len(self.p)), key=lambda x: self.p[x])
        r = self.p[i]
        start = (i - r) // 2
        return self.raw[start : start + r]

    def update(self, s):
        if s == self.raw:
            return False
        self.add(s)
        return True

    def delete(self):
        self.raw, self.t, self.p = "", "", []


def _test():
    m = Manacher()
    m.add("babad")
    assert m.query() in ("bab", "aba")
    m.add("cbbd")
    assert m.query() == "bb"
    assert m.update("a") and m.query() == "a"
    m.delete()
    assert m.query() == ""
    print("Manacher OK", Manacher("babad").query())


APPLICATION = """
- 回文半径数组一次扫完所有中心；比枚举中心扩展稳 O(n)。
- 敏感词变形、DNA 回文基序、面试 LC5。
- 动态插删字符维护回文 → 更难，需回文树(eertree)。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
