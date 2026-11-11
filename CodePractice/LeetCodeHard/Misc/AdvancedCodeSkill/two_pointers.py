"""
CodeSkill / 双指针
Template: query(pair_count|compress) / update / add / del
App: 有序两数和；同向滑窗基底。
"""


class TwoPointers:
    def __init__(self):
        self.a = []

    def add(self, x):
        if isinstance(x, (list, tuple)):
            self.a.extend(x)
        else:
            self.a.append(x)
        self.a.sort()

    def update(self, idx, val):
        if not (0 <= idx < len(self.a)):
            return False
        self.a[idx] = val
        self.a.sort()
        return True

    def delete(self, val):
        if val in self.a:
            self.a.remove(val)
            return True
        return False

    def query(self, mode="pair_sum", target=0):
        a = self.a
        if mode == "pair_sum":
            # 有序数组两数和 == target 的配对数（i<j）
            i, j, cnt = 0, len(a) - 1, 0
            while i < j:
                s = a[i] + a[j]
                if s == target:
                    cnt += 1
                    i += 1
                    j -= 1
                elif s < target:
                    i += 1
                else:
                    j -= 1
            return cnt
        if mode == "unique":
            # 原地去重后长度
            if not a:
                return 0
            w = 1
            for r in range(1, len(a)):
                if a[r] != a[w - 1]:
                    a[w] = a[r]
                    w += 1
            self.a = a[:w]
            return w
        raise ValueError(mode)


def _test():
    t = TwoPointers()
    t.add([1, 2, 3, 4, 5])
    assert t.query("pair_sum", 6) == 2  # 1+5, 2+4
    t.add(2)
    assert t.update(0, 0)
    assert t.delete(5)
    print("TwoPointers OK", t.query("unique"), t.a)


APPLICATION = """
- 对撞：有序两端；同向：快慢 / 滑窗右扩左缩。
- 三数和：枚举 i + 对撞，注意去重。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
