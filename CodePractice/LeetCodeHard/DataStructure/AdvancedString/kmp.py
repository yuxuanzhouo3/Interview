"""
1. KMP (Knuth-Morris-Pratt)
Template: query(match) / update(rebuild pattern) / add(pattern/text) / del
App: 单模式串匹配；grep 思想祖先；编辑器查找；LC 28 / 1392。
"""


class KMP:
    def __init__(self, pattern=""):
        self.pattern = pattern
        self.pi = self._prefix(pattern) if pattern else []

    @staticmethod
    def _prefix(s):
        n = len(s)
        pi = [0] * n
        for i in range(1, n):
            j = pi[i - 1]
            while j > 0 and s[j] != s[i]:
                j = pi[j - 1]
            if s[j] == s[i]:
                j += 1
            pi[i] = j
        return pi

    # ---------- add ----------
    def add(self, pattern):
        """设置/追加为新模式串（替换）。"""
        self.pattern = pattern
        self.pi = self._prefix(pattern)
        return self.pi

    # ---------- query ----------
    def query(self, text, first_only=False):
        """返回所有匹配起始下标。"""
        if not self.pattern:
            return []
        res, j = [], 0
        for i, ch in enumerate(text):
            while j > 0 and self.pattern[j] != ch:
                j = self.pi[j - 1]
            if self.pattern[j] == ch:
                j += 1
            if j == len(self.pattern):
                res.append(i - j + 1)
                if first_only:
                    return res
                j = self.pi[j - 1]
        return res

    def query_pi(self):
        return list(self.pi)

    # ---------- update ----------
    def update(self, pattern):
        if pattern == self.pattern:
            return False
        self.add(pattern)
        return True

    # ---------- del ----------
    def delete(self):
        self.pattern = ""
        self.pi = []


def _test():
    k = KMP()
    k.add("abab")
    assert k.query_pi() == [0, 0, 1, 2]
    assert k.query("ababaabab") == [0, 5]
    assert k.update("aa") and k.query("aabaa") == [0, 3]
    k.delete()
    assert k.query("abc") == []
    print("KMP OK", KMP("abab").query("ababaabab"))


APPLICATION = """
- 单模式精确匹配 O(n+m)；pi[] = 最长真前后缀。
- Java: 无标准 KMP，Pattern/Matcher 多为 NFA；面试手写 pi。
- vs BF O(nm)；vs Rabin-Karp 哈希；多模式 → Aho-Corasick。
- 应用: 日志关键字、DNA 子串、周期串判断（n % (n-pi[n-1])==0）。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
