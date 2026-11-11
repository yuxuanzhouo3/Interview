"""
7. 二分图冲突检测（奇偶关系 DSU）
App: LC 886 思想。
"""


class BipartiteDSU:
    """d[x]=0 与根同色，1 异色。"""

    def __init__(self, n):
        self.p = list(range(n))
        self.d = [0] * n

    def find(self, x):
        if self.p[x] != x:
            fa = self.p[x]
            self.p[x] = self.find(fa)
            self.d[x] ^= self.d[fa]
        return self.p[x]

    def update(self, a, b):
        """加「异色边」a-b；冲突返回 False。"""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return self.d[a] ^ self.d[b] == 1
        self.p[ra] = rb
        # d[a]^d[ra] ^ ? ^ d[b] = 1  → d[ra] = d[a]^d[b]^1
        self.d[ra] = self.d[a] ^ self.d[b] ^ 1
        return True

    def query(self, a, b):
        if self.find(a) != self.find(b):
            return None
        return self.d[a] ^ self.d[b]  # 0 same 1 diff

    def add(self, a, b):
        return self.update(a, b)

    def delete(self, a, b):
        return False


def possible_bipartition(n, dislikes):
    dsu = BipartiteDSU(n + 1)
    for a, b in dislikes:
        if not dsu.update(a, b):
            return False
    return True


def _test():
    assert possible_bipartition(4, [[1, 2], [1, 3], [2, 4]]) is True
    assert possible_bipartition(3, [[1, 2], [1, 3], [2, 3]]) is False
    print("dsu_bipartite OK")


APPLICATION = """
- 嫌恶边 = 必须异色；同根却同色 → 冲突。
- 也可用扩展域：x 与 y+n 合并。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
