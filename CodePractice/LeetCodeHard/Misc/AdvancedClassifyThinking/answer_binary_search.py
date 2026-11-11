"""
ClassifyThinking / 答案二分
Template: query(min_feasible) / update(check) / add(arr) / del
App: 单调判定；LC 410 / 875 / 1011 / 1482。
"""


class AnswerBinarySearch:
    def __init__(self):
        self.arr = []
        self.check = None  # fn(mid) -> bool，True 表示 mid 可行

    def add(self, x):
        if isinstance(x, (list, tuple)):
            self.arr.extend(x)
        else:
            self.arr.append(x)

    def update(self, check_fn):
        """设置单调判定函数。"""
        self.check = check_fn
        return True

    def delete(self, x=None):
        if x is None:
            self.arr.clear()
            return
        if x in self.arr:
            self.arr.remove(x)

    def query(self, lo, hi):
        """找最小可行 mid ∈ [lo, hi]；无解返回 None。"""
        if not self.check:
            raise ValueError("call update(check) first")
        ans = None
        while lo <= hi:
            mid = (lo + hi) // 2
            if self.check(mid):
                ans = mid
                hi = mid - 1
            else:
                lo = mid + 1
        return ans


def _test():
    # 分割数组最大和最小化：LC 410 简化，m=2
    a = AnswerBinarySearch()
    a.add([7, 2, 5, 10, 8])

    def can(limit):
        parts, cur = 1, 0
        for x in a.arr:
            if x > limit:
                return False
            if cur + x > limit:
                parts += 1
                cur = 0
            cur += x
        return parts <= 2

    a.update(can)
    assert a.query(max(a.arr), sum(a.arr)) == 18  # [7,2,5] | [10,8]
    a.delete(10)
    # 重新 check 仍引用 a.arr
    assert a.query(max(a.arr), sum(a.arr)) is not None
    print("AnswerBinarySearch OK", a.query(max(a.arr), sum(a.arr)))


APPLICATION = """
- 先证明 check(mid) 单调，再二分答案。
- 最小化可行 → 可行时收右；最大化可行 → 可行时扩左。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
