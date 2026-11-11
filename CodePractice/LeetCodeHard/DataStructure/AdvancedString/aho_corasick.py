"""
5. Aho-Corasick (多模式串)
Template: query(text hits) / update(rebuild) / add(pattern) / del(pattern rebuild)
App: 敏感词过滤、入侵检测、多关键字高亮；部分 WAF；多模式 grep。
"""

from collections import deque


class AhoCorasick:
    def __init__(self):
        self.goto = [{}]
        self.fail = [0]
        self.out = [[]]
        self.patterns = []
        self._built = False

    def add(self, pattern):
        pid = len(self.patterns)
        self.patterns.append(pattern)
        node = 0
        for ch in pattern:
            if ch not in self.goto[node]:
                self.goto[node][ch] = len(self.goto)
                self.goto.append({})
                self.fail.append(0)
                self.out.append([])
            node = self.goto[node][ch]
        self.out[node].append(pid)
        self._built = False
        return pid

    def _build_fail(self):
        # reset out to only terminal ids (fail merge re-applied)
        terminals = [list(x) for x in self.out]
        self.out = [list(x) for x in terminals]
        q = deque()
        for nxt in self.goto[0].values():
            q.append(nxt)
            self.fail[nxt] = 0
        while q:
            r = q.popleft()
            for ch, u in self.goto[r].items():
                q.append(u)
                v = self.fail[r]
                while v and ch not in self.goto[v]:
                    v = self.fail[v]
                self.fail[u] = self.goto[v].get(ch, 0)
                self.out[u] = terminals[u] + self.out[self.fail[u]]
        self._built = True

    def query(self, text):
        if not self._built:
            self._build_fail()
        node, hits = 0, []
        for i, ch in enumerate(text):
            while node and ch not in self.goto[node]:
                node = self.fail[node]
            node = self.goto[node].get(ch, 0)
            for pid in self.out[node]:
                hits.append((i, pid, self.patterns[pid]))
        return hits

    def update(self, pid, new_pattern):
        if not (0 <= pid < len(self.patterns)):
            return False
        pats = list(self.patterns)
        pats[pid] = new_pattern
        self._reset_from(pats)
        return True

    def delete(self, pid):
        if not (0 <= pid < len(self.patterns)):
            return False
        pats = [p for i, p in enumerate(self.patterns) if i != pid]
        self._reset_from(pats)
        return True

    def _reset_from(self, pats):
        self.__init__()
        for p in pats:
            self.add(p)


def _test():
    ac = AhoCorasick()
    ac.add("he")
    ac.add("she")
    ac.add("his")
    ac.add("hers")
    hits = ac.query("ushers")
    names = {p for *_, p in hits}
    assert "she" in names and "hers" in names
    assert ac.update(0, "her")
    assert ac.delete(2)  # drop former index carefully after update — patterns changed
    print("AC OK", [(i, p) for i, _, p in ac.query("ushers")])


APPLICATION = """
- 多模式 O(n + 命中数 + 模式总长)；fail = KMP 自动机推广。
- 业务: 敏感词、病毒特征初筛、代码搜索多关键字。
- Java: 无 JDK 内置；自研或 Lucene 分析链。
- vs 多次 KMP: 模式一多 AC 完胜。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
