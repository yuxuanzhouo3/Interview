"""
LC 895. Maximum Frequency Stack — Hard
Best: freq map + freq→stack — O(1)
"""

from collections import defaultdict


class FreqStack:
    def __init__(self):
        self.freq = defaultdict(int)
        self.group = defaultdict(list)
        self.maxf = 0

    def push(self, val: int) -> None:
        f = self.freq[val] + 1
        self.freq[val] = f
        if f > self.maxf:
            self.maxf = f
        self.group[f].append(val)

    def pop(self) -> int:
        val = self.group[self.maxf].pop()
        self.freq[val] -= 1
        if not self.group[self.maxf]:
            self.maxf -= 1
        return val


def _test():
    fs = FreqStack()
    for x in [5, 7, 5, 7, 4, 5]:
        fs.push(x)
    assert fs.pop() == 5
    assert fs.pop() == 7
    assert fs.pop() == 5
    assert fs.pop() == 4
    print("LC 895 OK")


if __name__ == "__main__":
    _test()
