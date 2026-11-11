"""
7. Suffix Automaton (SAM)
Template: query(occur / contains) / update(rebuild) / add(char|string) / del(rebuild)
App: 子串出现次数、本质不同子串数、最长公共子串；竞赛神器。
"""


class State:
    __slots__ = ("next", "link", "len", "occ")

    def __init__(self):
        self.next = {}
        self.link = -1
        self.len = 0
        self.occ = 0


class SAM:
    def __init__(self):
        self.st = [State()]
        self.last = 0
        self.s = []

    def add(self, chars):
        for ch in chars:
            self._extend(ch)
            self.s.append(ch)
        return len(self.st)

    def _extend(self, ch):
        cur = len(self.st)
        self.st.append(State())
        self.st[cur].len = self.st[self.last].len + 1
        self.st[cur].occ = 1
        p = self.last
        while p != -1 and ch not in self.st[p].next:
            self.st[p].next[ch] = cur
            p = self.st[p].link
        if p == -1:
            self.st[cur].link = 0
        else:
            q = self.st[p].next[ch]
            if self.st[p].len + 1 == self.st[q].len:
                self.st[cur].link = q
            else:
                clone = len(self.st)
                self.st.append(State())
                self.st[clone].len = self.st[p].len + 1
                self.st[clone].next = dict(self.st[q].next)
                self.st[clone].link = self.st[q].link
                self.st[clone].occ = 0
                while p != -1 and self.st[p].next.get(ch) == q:
                    self.st[p].next[ch] = clone
                    p = self.st[p].link
                self.st[q].link = self.st[cur].link = clone
        self.last = cur

    def _occ_array(self):
        occ = [st.occ for st in self.st]
        order = sorted(range(len(self.st)), key=lambda i: self.st[i].len, reverse=True)
        for v in order:
            link = self.st[v].link
            if link != -1:
                occ[link] += occ[v]
        return occ

    def query(self, pattern=None, mode="contains"):
        if pattern is None and mode == "distinct":
            return sum(
                self.st[i].len - self.st[self.st[i].link].len for i in range(1, len(self.st))
            )
        if pattern is None:
            return "".join(self.s)
        node = 0
        for ch in pattern:
            if ch not in self.st[node].next:
                return False if mode == "contains" else 0
            node = self.st[node].next[ch]
        if mode == "contains":
            return True
        return self._occ_array()[node]

    def update(self, s):
        self.__init__()
        self.add(s)
        return True

    def delete(self):
        self.__init__()


def _test():
    sam = SAM()
    sam.add("abacaba")
    assert sam.query("aba") is True
    assert sam.query("abc") is False
    assert sam.query("aba", mode="occ") >= 2
    assert sam.query(mode="distinct") > 0
    sam.update("aaa")
    assert sam.query("aa") is True
    assert sam.query("aa", mode="occ") == 2
    sam.delete()
    assert sam.query("a") is False
    print("SAM OK", SAM().add("x") or "built")


APPLICATION = """
- 认所有后缀的最小 DFA；子串查询/出现次数/本质不同子串。
- 基因组、竞赛字符串题；工程上 AC/倒排更常见。
- 大小 O(n) 状态，转移 O(n)。
"""

if __name__ == "__main__":
    s = SAM()
    s.add("abacaba")
    print("SAM OK occ(aba)=", s.query("aba", mode="occ"), "distinct=", s.query(mode="distinct"))
    _test()
    print(APPLICATION)
