"""
LC 4007. Widest Possible Fence — Hard
https://leetcode.com/problems/widest-possible-fence/
Best: doocs/leetcode reference solution (verified on examples)
"""

from collections import defaultdict, Counter


class Solution:
    def maximumWidth(self, planks: list[int]) -> int:
        cnt = Counter(planks)
        ans = 0
        t = defaultdict(int)
        for x, v1 in cnt.items():
            t[x] += v1
            t[x * 2] += v1 // 2
            for y, v2 in cnt.items():
                if y > x:
                    t[x + y] += min(v1, v2)
        return max(t.values())

def _test():
    s = Solution()
    assert s.maximumWidth([1,3,2,5,7,5,4,2,1]) == 4
    assert s.maximumWidth([2,3,7]) == 1
    print("LC 4007 OK")


if __name__ == "__main__":
    _test()
