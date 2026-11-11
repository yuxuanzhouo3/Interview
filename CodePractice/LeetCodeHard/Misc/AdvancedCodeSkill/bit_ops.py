"""
CodeSkill / 位运算
Template: query(count|subset|lowbit) / update / add / del
App: 状压预处理；树状数组 lowbit；LC 191 / 338。
"""


class BitOps:
    def __init__(self):
        self.mask = 0

    def add(self, bit):
        """置 1 某比特（0-based）。"""
        self.mask |= 1 << bit

    def update(self, bit, on=True):
        if on:
            self.mask |= 1 << bit
        else:
            self.mask &= ~(1 << bit)
        return True

    def delete(self, bit):
        self.mask &= ~(1 << bit)

    def query(self, mode="count", x=None):
        m = self.mask if x is None else x
        if mode == "count":
            return bin(m).count("1")
        if mode == "lowbit":
            return m & -m
        if mode == "subsets":
            # 枚举 m 的非空子集
            res, sub = [], m
            while sub:
                res.append(sub)
                sub = (sub - 1) & m
            return res
        raise ValueError(mode)


def _test():
    b = BitOps()
    b.add(0)
    b.add(2)
    assert b.query("count") == 2
    assert b.query("lowbit") == 1
    assert b.update(0, False)
    assert b.query("count") == 1
    b.add(1)
    subs = b.query("subsets")
    assert 0b110 in subs or 6 in subs
    b.delete(2)
    print("BitOps OK", b.query("count"), b.query("subsets"))


APPLICATION = """
- `i += i & -i` / `i -= i & -i`：BIT 上下跳。
- 子集枚举：`sub = (sub - 1) & mask`。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
