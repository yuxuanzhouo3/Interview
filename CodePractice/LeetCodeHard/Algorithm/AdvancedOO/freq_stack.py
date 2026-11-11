"""
4. Maximum Frequency Stack
Template: query(top via pop) / update / add(push) / del(pop)
App: LC 895。
"""

from collections import defaultdict


class FreqStack:
    def __init__(self):
        self.freq = defaultdict(int)
        self.group = defaultdict(list)  # freq -> stack of vals
        self.maxf = 0

    def add(self, val):
        self.push(val)

    def push(self, val):
        self.freq[val] += 1
        f = self.freq[val]
        self.maxf = max(self.maxf, f)
        self.group[f].append(val)

    def update(self, val):
        """再压入一次视为升温。"""
        self.push(val)
        return True

    def query(self):
        if self.maxf == 0:
            return None
        return self.group[self.maxf][-1]

    def delete(self):
        return self.pop()

    def pop(self):
        val = self.group[self.maxf].pop()
        self.freq[val] -= 1
        if not self.group[self.maxf]:
            self.maxf -= 1
        return val


def _test():
    s = FreqStack()
    for x in [5, 7, 5, 7, 4, 5]:
        s.push(x)
    assert s.pop() == 5
    assert s.pop() == 7
    assert s.pop() == 5
    assert s.pop() == 4
    print("FreqStack OK")


APPLICATION = """
- 同频后进先出；始终弹 maxf 组栈顶。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
