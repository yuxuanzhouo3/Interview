"""
CodeSkill / 二分上下界模板
Template: query(lower|upper|first_true) / update(arr) / add / del
App: 有序数组边界；答案二分的 lo/hi 写法。
"""


class BinarySearch:
    def __init__(self):
        self.a = []

    def add(self, x):
        if isinstance(x, (list, tuple)):
            self.a.extend(x)
            self.a.sort()
        else:
            # 有序插入
            lo, hi = 0, len(self.a)
            while lo < hi:
                mid = (lo + hi) // 2
                if self.a[mid] < x:
                    lo = mid + 1
                else:
                    hi = mid
            self.a.insert(lo, x)

    def update(self, idx, val):
        if not (0 <= idx < len(self.a)):
            return False
        self.a.pop(idx)
        self.add(val)
        return True

    def delete(self, val):
        # 删一个等于 val 的
        i = self.query("lower", val)
        if i < len(self.a) and self.a[i] == val:
            self.a.pop(i)
            return True
        return False

    def query(self, mode, target):
        a = self.a
        if mode == "lower":  # first >= target
            lo, hi = 0, len(a)
            while lo < hi:
                mid = (lo + hi) // 2
                if a[mid] < target:
                    lo = mid + 1
                else:
                    hi = mid
            return lo
        if mode == "upper":  # first > target
            lo, hi = 0, len(a)
            while lo < hi:
                mid = (lo + hi) // 2
                if a[mid] <= target:
                    lo = mid + 1
                else:
                    hi = mid
            return lo
        raise ValueError(mode)


def _test():
    b = BinarySearch()
    b.add([1, 3, 3, 5, 7])
    assert b.query("lower", 3) == 1
    assert b.query("upper", 3) == 3
    assert b.update(0, 4)
    assert b.delete(3)
    print("BinarySearch OK", b.a)


APPLICATION = """
- lower_bound / upper_bound 是一切二分的底座。
- 答案二分：把 arr 换成 check(mid) 的单调性即可。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
