"""
2. Z-Algorithm (扩展 KMP)
Template: query(Z-box / match) / update(rebuild) / add(string) / del
App: 与字符串本身匹配的各后缀；求周期；替代部分 KMP 场景。
Z[i] = s 与 s[i:] 的最长公共前缀长度。
"""


class ZAlgorithm:
    def __init__(self, s=""):
        self.s = s
        self.z = self._build(s) if s else []

    @staticmethod
    def _build(s):
        n = len(s)
        z = [0] * n
        l = r = 0
        for i in range(1, n):
            if i <= r:
                z[i] = min(r - i + 1, z[i - l])
            while i + z[i] < n and s[z[i]] == s[i + z[i]]:
                z[i] += 1
            if i + z[i] - 1 > r:
                l, r = i, i + z[i] - 1
        z[0] = n
        return z

    def add(self, s):
        self.s = s
        self.z = self._build(s)
        return self.z

    def query(self, i=None):
        """query() 全部 Z；query(i) 单点；或 match(pattern, text)。"""
        if i is None:
            return list(self.z)
        return self.z[i] if 0 <= i < len(self.z) else None

    def match(self, pattern, text):
        """pattern 在 text 中的出现位置（用 Z 拼接技巧）。"""
        comb = pattern + "#" + text
        z = self._build(comb)
        m = len(pattern)
        return [i - m - 1 for i in range(m + 1, len(comb)) if z[i] == m]

    def update(self, s):
        if s == self.s:
            return False
        self.add(s)
        return True

    def delete(self):
        self.s, self.z = "", []


def _test():
    z = ZAlgorithm()
    z.add("aabcaabxaaaz")
    assert z.query(0) == 12
    assert z.query(3) == 0  # 'c...'
    assert z.match("aab", "xaabyaab") == [1, 5]
    assert z.update("aaa") and z.query() == [3, 2, 1]
    z.delete()
    assert z.query() == []
    print("Z OK", ZAlgorithm("aabcaabxaaaz").query())


APPLICATION = """
- O(n) 构建；常与 KMP 互换做「自身匹配」。
- 字符串周期、最长回文相关预处理、竞赛模板题。
- Java 无内置；Guava/自写。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
