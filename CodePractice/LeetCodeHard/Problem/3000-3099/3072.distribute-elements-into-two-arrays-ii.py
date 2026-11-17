"""
LC 3072. Distribute Elements Into Two Arrays II — Hard
https://leetcode.com/problems/distribute-elements-into-two-arrays-ii/
Best: Fenwick rank count — O(n log n)
"""

from typing import List


class Solution:
    def resultArray(self, nums: List[int]) -> List[int]:
        # BIT for counts of values
        vals = sorted(set(nums))
        rank = {v: i + 1 for i, v in enumerate(vals)}
        m = len(vals)

        class BIT:
            def __init__(self):
                self.b = [0] * (m + 1)
            def add(self, i, d=1):
                while i <= m:
                    self.b[i] += d
                    i += i & -i
            def sum(self, i):
                s = 0
                while i:
                    s += self.b[i]
                    i -= i & -i
                return s
            def greater(self, v):
                # count ranks > rank[v]
                r = rank[v]
                return self.sum(m) - self.sum(r)

        a, b = [nums[0]], [nums[1]]
        ba, bb = BIT(), BIT()
        ba.add(rank[nums[0]])
        bb.add(rank[nums[1]])
        for x in nums[2:]:
            ga, gb = ba.greater(x), bb.greater(x)
            if ga > gb or (ga == gb and len(a) <= len(b)):
                a.append(x)
                ba.add(rank[x])
            else:
                b.append(x)
                bb.add(rank[x])
        return a + b


def _test():
    s = Solution()
    assert s.resultArray([2, 1, 3, 3]) == [2, 3, 1, 3]
    assert s.resultArray([5, 4, 3, 8]) == [5, 3, 4, 8]
    print("LC 3072 OK")


if __name__ == "__main__":
    _test()

