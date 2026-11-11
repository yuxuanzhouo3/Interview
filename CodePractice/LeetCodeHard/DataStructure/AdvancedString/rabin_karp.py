"""
3. Rabin-Karp (Rolling Hash)
Template: query(subhash / find) / update(point char) / add(append) / del(pop)
App: 子串相等判定、多模式初筛、差分/剽窃检测；LC 187 / 1044 思想。
"""


class RollingHash:
    def __init__(self, s="", base=911382323, mod=10**9 + 7):
        self.base = base
        self.mod = mod
        self.s = []
        self.pref = [0]  # pref[i] = hash(s[:i])
        self.pow = [1]   # pow[i] = base^i
        if s:
            self.add(s)

    def add(self, chars):
        """追加字符（支持 str）。"""
        for ch in chars:
            self.s.append(ch)
            self.pow.append(self.pow[-1] * self.base % self.mod)
            self.pref.append((self.pref[-1] * self.base + ord(ch)) % self.mod)

    def query(self, l, r=None):
        """
        query(l, r): hash of s[l:r]  (半开区间)
        query(pattern): 在当前串中找 pattern 所有起点
        """
        if isinstance(l, str):
            return self.find(l)
        if r is None:
            r = l + 1
        return (self.pref[r] - self.pref[l] * self.pow[r - l]) % self.mod

    def find(self, pattern):
        m, n = len(pattern), len(self.s)
        if m == 0 or m > n:
            return []
        ph = 0
        for ch in pattern:
            ph = (ph * self.base + ord(ch)) % self.mod
        out = []
        for i in range(n - m + 1):
            if self.query(i, i + m) == ph and "".join(self.s[i : i + m]) == pattern:
                out.append(i)
        return out

    def update(self, i, ch):
        """单点改字符 → 重建后缀前缀（教学 O(n)；竞赛可线段树哈希）。"""
        if not (0 <= i < len(self.s)):
            return False
        self.s[i] = ch
        s = "".join(self.s)
        self.s, self.pref, self.pow = [], [0], [1]
        self.add(s)
        return True

    def delete(self, n=1):
        """从末尾删 n 个字符。"""
        n = min(n, len(self.s))
        for _ in range(n):
            self.s.pop()
            self.pref.pop()
            self.pow.pop()


def _test():
    rh = RollingHash()
    rh.add("abracadabra")
    assert rh.find("abra") == [0, 7]
    h1 = rh.query(0, 4)
    assert rh.query(7, 11) == h1
    assert rh.update(0, "x")
    assert rh.find("abra") == [7]
    rh.delete(4)  # drop 'abra' tail-ish — last 4 of "xbracadabra"
    assert len(rh.s) == 7
    print("RabinKarp OK", RollingHash("abracadabra").find("abra"))


APPLICATION = """
- 平均 O(n+m)；需取模防撞，重要场景双哈希。
- Redis/搜索: 布隆+哈希指纹；文件去重 CDC 用滚动哈希变体（Rabin fingerprint）。
- Java: 可用自写；MessageDigest 太重不适合子串滚动。
- vs KMP: RK 易扩展多模式/二维；KMP 最坏保证更好。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
