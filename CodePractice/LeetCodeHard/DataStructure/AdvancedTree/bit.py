"""
8. BIT / Fenwick Tree (Binary Indexed Tree)
Template: query(prefix/range) / update / add / del(update -val)
App: 竞赛高频；前缀和动态维护；逆序对；LeetCode 307/315 类。
下标通常 1-based。
"""


class BIT:
    def __init__(self, n):
        self.n = n
        self.tree = [0] * (n + 1)

    # ---------- add / update point ----------
    def add(self, i, delta):
        """下标 i (1-based) 增加 delta。"""
        while i <= self.n:
            self.tree[i] += delta
            i += i & -i

    def update(self, i, new_val, old_val=None):
        """把位置 i 设为 new_val；若未知 old，需外部维护原数组。"""
        if old_val is None:
            raise ValueError("BIT.update needs old_val (or maintain arr[])")
        self.add(i, new_val - old_val)

    # ---------- query ----------
    def query_prefix(self, i):
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & -i
        return s

    def query(self, l, r):
        """区间和 [l, r]，1-based inclusive。"""
        if r < l:
            return 0
        return self.query_prefix(r) - self.query_prefix(l - 1)

    # ---------- del ----------
    def delete(self, i, old_val):
        """逻辑删除：减去旧值。"""
        self.add(i, -old_val)


def _test():
    arr = [0, 1, 2, 3, 4, 5]  # 1-based mirror
    bit = BIT(5)
    for i in range(1, 6):
        bit.add(i, arr[i])
    assert bit.query(1, 5) == 15
    assert bit.query_prefix(3) == 6
    bit.update(3, 10, old_val=arr[3])
    arr[3] = 10
    assert bit.query(1, 5) == 22
    bit.delete(2, old_val=arr[2])
    arr[2] = 0
    assert bit.query(1, 5) == 20
    print("BIT OK", bit.query(1, 5))


APPLICATION = """
- 竞赛: 动态前缀和、逆序对、频率树状数组。
- vs Segment Tree: BIT 常数小、写短；线段树更易做区间最值/懒标记复杂操作。
- 工业: 广告计费实时累加、简单监控计数；复杂区间操作用线段树/稀疏表。
- LeetCode: 307 Range Sum Query - Mutable, 315 Count of Smaller Numbers.
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
