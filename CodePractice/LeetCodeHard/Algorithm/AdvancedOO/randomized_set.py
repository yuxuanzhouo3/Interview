"""
5. Insert Delete GetRandom O(1)
Template: query(random/contains) / update / add / del
App: LC 380。
"""

import random


class RandomizedSet:
    def __init__(self):
        self.arr = []
        self.idx = {}

    def add(self, val):
        if val in self.idx:
            return False
        self.idx[val] = len(self.arr)
        self.arr.append(val)
        return True

    def update(self, old, new):
        if not self.delete(old):
            return False
        return self.add(new)

    def query(self, val=None):
        if val is None:
            return self.getRandom()
        return val in self.idx

    def delete(self, val):
        if val not in self.idx:
            return False
        i = self.idx[val]
        last = self.arr[-1]
        self.arr[i] = last
        self.idx[last] = i
        self.arr.pop()
        del self.idx[val]
        return True

    def getRandom(self):
        return random.choice(self.arr)


def _test():
    random.seed(0)
    s = RandomizedSet()
    assert s.add(1) and s.add(2) and not s.add(1)
    assert s.delete(1) and not s.query(1)
    assert s.query(2)
    assert s.getRandom() == 2
    print("RandomizedSet OK")


APPLICATION = """
- 删：与末尾交换再 pop，同步哈希下标。
- LC 381 允许多重：值→下标集合。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
