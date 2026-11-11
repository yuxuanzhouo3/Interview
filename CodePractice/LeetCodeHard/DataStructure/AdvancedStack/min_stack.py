"""
AdvancedStack / MinStack
Template: query(top/getMin) / update(via push new) / add(push) / del(pop)
App: LC 155；需要 O(1) 取最小值的撤销栈。
"""


class MinStack:
    def __init__(self):
        self.st = []
        self.mins = []

    def add(self, x):
        self.st.append(x)
        self.mins.append(x if not self.mins else min(x, self.mins[-1]))

    push = add

    def delete(self):
        if not self.st:
            return None
        self.mins.pop()
        return self.st.pop()

    pop = delete

    def query(self, mode="top"):
        if not self.st:
            return None
        return self.st[-1] if mode == "top" else self.mins[-1]

    def getMin(self):
        return self.query("min")

    def update(self, x):
        """弹栈顶再压入新值。"""
        if not self.st:
            self.add(x)
            return
        self.delete()
        self.add(x)


def _test():
    s = MinStack()
    s.add(3)
    s.add(1)
    s.add(2)
    assert s.query("top") == 2 and s.getMin() == 1
    s.delete()
    assert s.getMin() == 1
    s.update(0)
    assert s.getMin() == 0
    print("MinStack OK", s.getMin())


APPLICATION = """
- 编辑器撤销且跟踪最小分数；游戏 buff 栈。
- 对称: MaxStack；双栈实现队列。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
