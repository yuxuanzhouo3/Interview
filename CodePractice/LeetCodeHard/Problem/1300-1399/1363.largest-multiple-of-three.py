"""
LC 1363. Largest Multiple of Three — Hard
https://leetcode.com/problems/largest-multiple-of-three/
Best: Digit count + remove min by mod — O(n)
"""

from typing import List


class Solution:
    def largestMultipleOfThree(self, digits: List[int]) -> str:
        cnt = [0] * 10
        s = 0
        for d in digits:
            cnt[d] += 1
            s += d
        def remove_one(mod):
            for d in range(1, 10):
                if d % 3 == mod and cnt[d]:
                    cnt[d] -= 1
                    return True
            return False
        def remove_two(mod):
            return remove_one(mod) and remove_one(mod)
        r = s % 3
        if r:
            if not remove_one(r):
                remove_two(3 - r)
        parts = []
        for d in range(9, -1, -1):
            parts.append(str(d) * cnt[d])
        ans = "".join(parts)
        if not ans:
            return ""
        if ans[0] == "0":
            return "0"
        return ans


def _test():
    s = Solution()
    assert s.largestMultipleOfThree([8, 1, 9]) == "981"
    assert s.largestMultipleOfThree([8, 6, 7, 1, 0]) == "8760"
    assert s.largestMultipleOfThree([1]) == ""
    print("LC 1363 OK")


if __name__ == "__main__":
    _test()
