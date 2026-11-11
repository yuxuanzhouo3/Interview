"""
AdvancedStack / Monotonic Stack
Template: query(top/next greater) / update(recompute via push) / add(push) / del(pop)
App: 柱状图最大矩形、接雨水、下一个更大元素。
"""


class MonotonicStack:
    """默认严格递减栈（栈内从底到顶递减）→ 找下一个更大。"""

    def __init__(self, decreasing=True):
        self.decreasing = decreasing
        self.st = []  # values or (index, value)

    def add(self, x):
        """push with mono maintain; returns popped list."""
        popped = []
        if self.decreasing:
            while self.st and self.st[-1] <= x:
                popped.append(self.st.pop())
        else:
            while self.st and self.st[-1] >= x:
                popped.append(self.st.pop())
        self.st.append(x)
        return popped

    def delete(self):
        return self.st.pop() if self.st else None

    def query(self):
        return self.st[-1] if self.st else None

    def update(self, new_top):
        """替换栈顶（少见）；保持单调可能需重建。"""
        if not self.st:
            self.add(new_top)
            return
        self.st.pop()
        self.add(new_top)


def next_greater(nums):
    n = len(nums)
    ans = [-1] * n
    st = []  # indices, values increasing from bottom? store idx, mono decreasing values
    for i, x in enumerate(nums):
        while st and nums[st[-1]] < x:
            ans[st.pop()] = x
        st.append(i)
    return ans


def _test():
    assert next_greater([2, 1, 2, 4, 3]) == [4, 2, 4, -1, -1]
    ms = MonotonicStack(decreasing=True)
    ms.add(5)
    ms.add(3)
    assert ms.query() == 3
    popped = ms.add(4)  # pops 3
    assert popped == [3] and ms.query() == 4
    ms.delete()
    assert ms.query() == 5
    print("MonotonicStack OK", next_greater([2, 1, 2, 4, 3]))


APPLICATION = """
- LC 84 Largest Rectangle, 42 Trapping Rain Water, 496 Next Greater.
- 行情「下一冲高价」、渲染高度条、表达式温度等待天数。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
