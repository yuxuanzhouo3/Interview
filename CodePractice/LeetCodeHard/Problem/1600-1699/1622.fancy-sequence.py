"""
LC 1622. Fancy Sequence — Hard
https://leetcode.com/problems/fancy-sequence/
Best: Lazy affine transform + modular inverse — O(1) amortized
"""

MOD = 10**9 + 7


class Fancy:
    def __init__(self):
        self.a = []
        self.add = 0
        self.mul = 1

    def append(self, val: int) -> None:
        # store raw so that getIndex applies current mul/add:  raw*mul+add = val
        # => raw = (val - add) * inv(mul)
        inv = pow(self.mul, MOD - 2, MOD)
        self.a.append((val - self.add) * inv % MOD)

    def addAll(self, inc: int) -> None:
        self.add = (self.add + inc) % MOD

    def multAll(self, m: int) -> None:
        self.mul = self.mul * m % MOD
        self.add = self.add * m % MOD

    def getIndex(self, idx: int) -> int:
        if idx >= len(self.a):
            return -1
        return (self.a[idx] * self.mul + self.add) % MOD


def _test():
    f = Fancy()
    f.append(2)
    f.addAll(3)
    f.append(7)
    f.multAll(2)
    assert f.getIndex(0) == 10
    f.addAll(3)
    f.append(10)
    f.multAll(2)
    assert f.getIndex(0) == 26
    assert f.getIndex(1) == 34
    assert f.getIndex(2) == 20
    print("LC 1622 OK")


if __name__ == "__main__":
    _test()
