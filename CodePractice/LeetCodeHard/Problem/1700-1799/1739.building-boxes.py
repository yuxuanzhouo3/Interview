"""
LC 1739. Building Boxes — Hard
https://leetcode.com/problems/building-boxes/
Best: math tetrahedral packing — O(n^(1/3))
"""


class Solution:
    def minimumBoxes(self, n: int) -> int:
        # fill complete layers 1+(1+2)+...+(1+..+k)
        placed, k = 0, 1
        while placed + k * (k + 1) // 2 <= n:
            placed += k * (k + 1) // 2
            k += 1
        k -= 1
        ans = k * (k + 1) // 2  # ground boxes of full pyramid
        # place remainder with extra ground rows of size 1,2,3,...
        extra = 1
        while placed < n:
            ans += 1
            placed += extra
            extra += 1
        return ans


def _test():
    s = Solution()
    assert s.minimumBoxes(3) == 3
    assert s.minimumBoxes(4) == 3
    assert s.minimumBoxes(10) == 6
    print("LC 1739 OK")


if __name__ == "__main__":
    _test()
