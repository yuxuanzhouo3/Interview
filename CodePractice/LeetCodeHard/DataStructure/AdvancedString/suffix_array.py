"""
6. Suffix Array + LCP
Template: query(sa/lcp/contains) / update(rebuild) / add(string) / del
App: 子串搜索、最长重复子串、异位语；早期搜索引擎索引；LC 1044。
简版: 倍增 O(n log^2 n)。
"""


class SuffixArray:
    def __init__(self, s=""):
        self.s = s
        self.sa = []
        self.rank = []
        self.lcp = []
        if s:
            self.add(s)

    def add(self, s):
        self.s = s
        self.sa = self._build_sa(s)
        self.rank = [0] * len(s)
        for i, p in enumerate(self.sa):
            self.rank[p] = i
        self.lcp = self._build_lcp(s, self.sa, self.rank)
        return self.sa

    @staticmethod
    def _build_sa(s):
        n = len(s)
        sa = list(range(n))
        ra = [ord(c) for c in s]
        k = 1
        while k < n:
            sa.sort(key=lambda i: (ra[i], ra[i + k] if i + k < n else -1))
            nra = [0] * n
            for i in range(1, n):
                a, b = sa[i - 1], sa[i]
                nra[b] = nra[a] + (
                    (ra[a], ra[a + k] if a + k < n else -1)
                    != (ra[b], ra[b + k] if b + k < n else -1)
                )
            ra = nra
            if ra[sa[-1]] == n - 1:
                break
            k <<= 1
        return sa

    @staticmethod
    def _build_lcp(s, sa, rank):
        n = len(s)
        lcp = [0] * n
        h = 0
        for i in range(n):
            r = rank[i]
            if r == 0:
                continue
            j = sa[r - 1]
            while i + h < n and j + h < n and s[i + h] == s[j + h]:
                h += 1
            lcp[r] = h
            h = max(h - 1, 0)
        return lcp

    def query(self, what="sa"):
        if what == "sa":
            return list(self.sa)
        if what == "lcp":
            return list(self.lcp)
        if what == "lrs":  # longest repeated substring
            if not self.lcp:
                return ""
            i = max(range(len(self.lcp)), key=lambda x: self.lcp[x])
            length = self.lcp[i]
            return self.s[self.sa[i] : self.sa[i] + length]
        # substring search
        return self.contains(what)

    def contains(self, pattern):
        """二分 SA，返回是否存在 / 任一起点。"""
        if not self.s or not pattern:
            return None
        lo, hi = 0, len(self.sa)
        n, m = len(self.s), len(pattern)
        while lo < hi:
            mid = (lo + hi) // 2
            i = self.sa[mid]
            sub = self.s[i : min(i + m, n)]
            if sub < pattern:
                lo = mid + 1
            else:
                hi = mid
        if lo < len(self.sa):
            i = self.sa[lo]
            if self.s[i : i + m] == pattern:
                return i
        return None

    def update(self, s):
        if s == self.s:
            return False
        self.add(s)
        return True

    def delete(self):
        self.s, self.sa, self.rank, self.lcp = "", [], [], []


def _test():
    sa = SuffixArray()
    sa.add("banana")
    assert sa.query("ana") == 1 or sa.query("ana") == 3
    assert "ana" in sa.query("lrs") or sa.query("lrs") in ("ana", "an", "a", "n")
    # banana LRS is "ana"
    assert sa.query("lrs") == "ana"
    assert sa.update("aaaa") and sa.query("lrs") == "aaa"
    sa.delete()
    assert sa.query("sa") == []
    print("SuffixArray OK", SuffixArray("banana").query("sa"), "LRS", SuffixArray("banana").query("lrs"))


APPLICATION = """
- 子串查询、最长重复/公共子串（拼接+不同颜色 SA）。
- 现代引擎更多倒排+Trie/FST（Lucene）；SA 仍是竞赛/基因组标配。
- vs Suffix Tree: SA 更省空间；SAM 更强联机。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
