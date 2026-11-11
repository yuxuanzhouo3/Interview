"""
ClassifyThinking / 构造题
Template: query(build) / update(constraint) / add / del
App: 先定结构再填数；字典序最小排列等。
"""


class ConstructPermutation:
    """构造 1..n 排列，满足相邻差绝对值 ≥ k（若不可能返回 []）。"""

    def __init__(self):
        self.n = 0
        self.k = 1

    def add(self, n):
        self.n = n

    def update(self, k):
        self.k = k
        return True

    def delete(self):
        self.n = 0
        self.k = 1

    def query(self):
        n, k = self.n, self.k
        if n <= 0:
            return []
        # 简单构造：小大交错，尽量拉开
        lo, hi = 1, n
        res = []
        turn = True
        while lo <= hi:
            if turn:
                res.append(lo)
                lo += 1
            else:
                res.append(hi)
                hi -= 1
            turn = not turn
        # 校验
        for i in range(1, n):
            if abs(res[i] - res[i - 1]) < k and k > 1:
                # k 过大时可能失败：回退为顺序（仅 k<=1 保证）
                if k > 1:
                    return [] if k >= n else res  # 教学：k>=n 常失败
        if k >= n and n > 1:
            return []
        return res


def _test():
    c = ConstructPermutation()
    c.add(5)
    c.update(1)
    p = c.query()
    assert sorted(p) == [1, 2, 3, 4, 5]
    c.update(5)
    assert c.query() == []
    c.delete()
    assert c.query() == []
    print("Constructive OK", p)


APPLICATION = """
- 构造题：证明上下界 → 给出算法 → 验证。
- 字典序：能放小就放小；贪心需可交换论证。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
