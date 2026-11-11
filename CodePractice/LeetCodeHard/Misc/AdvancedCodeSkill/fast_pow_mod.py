"""
CodeSkill / 快速幂
Template: query(pow) / update(mod) / add(base) / del
App: LC 50 / 372；模意义下乘法逆元（费马，模质数）。
"""


class FastPow:
    def __init__(self, mod=10**9 + 7):
        self.base = 1
        self.mod = mod

    def add(self, base):
        self.base = base

    def update(self, mod):
        self.mod = mod
        return True

    def delete(self):
        self.base = 1

    def query(self, exp):
        return self.pow(self.base, exp, self.mod)

    @staticmethod
    def pow(x, n, mod=None):
        res = 1
        x = x if mod is None else x % mod
        while n > 0:
            if n & 1:
                res = res * x if mod is None else (res * x) % mod
            x = x * x if mod is None else (x * x) % mod
            n >>= 1
        return res

    def inv(self, x):
        """模质数逆元。"""
        return self.pow(x, self.mod - 2, self.mod)


def _test():
    f = FastPow()
    f.add(2)
    assert f.query(10) == 1024
    assert f.update(10**9 + 7)
    assert f.query(10) == 1024
    assert f.inv(2) * 2 % f.mod == 1
    f.delete()
    assert f.query(5) == 1
    print("FastPow OK", FastPow.pow(3, 5))


APPLICATION = """
- 二分指数：奇数乘入答案，底平方，指数右移。
- a/b mod p = a * b^(p-2) （p 质数）。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
