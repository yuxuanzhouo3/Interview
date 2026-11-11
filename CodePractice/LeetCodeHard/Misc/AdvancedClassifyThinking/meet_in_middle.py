"""
ClassifyThinking / Meet in the Middle（折半枚举）
Template: query(count/exist) / update(target) / add / del
App: n≤40 子集和；指数减半 2^(n/2)。
"""


class MeetInMiddle:
    def __init__(self):
        self.nums = []
        self.target = 0

    def add(self, x):
        if isinstance(x, (list, tuple)):
            self.nums.extend(x)
        else:
            self.nums.append(x)

    def update(self, target):
        self.target = target
        return True

    def delete(self, x=None):
        if x is None:
            self.nums.clear()
            return
        if x in self.nums:
            self.nums.remove(x)

    def _subset_sums(self, arr):
        sums = [0]
        for v in arr:
            sums += [s + v for s in sums]
        return sums

    def query(self, mode="exist"):
        n = len(self.nums)
        left = self._subset_sums(self.nums[: n // 2])
        right = self._subset_sums(self.nums[n // 2 :])
        rset = set(right)
        if mode == "exist":
            return any(self.target - s in rset for s in left)
        # count
        from collections import Counter

        rc = Counter(right)
        return sum(rc[self.target - s] for s in left)


def _test():
    m = MeetInMiddle()
    m.add([1, 2, 3, 4, 5, 6])
    m.update(11)
    assert m.query("exist")
    assert m.query("count") >= 1
    m.delete(6)
    m.update(100)
    assert not m.query("exist")
    print("MeetInMiddle OK", m.query("count"))


APPLICATION = """
- 指数 2^n → 2·2^(n/2)；半边排序可双指针求最接近。
- 与状压分界：n≤20 可状压转移；n≤40 偏折半。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
