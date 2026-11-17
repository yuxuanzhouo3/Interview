"""
LC 3821. Find Nth Smallest Integer With K One Bits — Hard
https://leetcode.com/problems/find-nth-smallest-integer-with-k-one-bits/
Best: doocs/leetcode reference solution (verified on examples)
"""


mx = 50
c = [[0] * (mx + 1) for _ in range(mx)]
for i in range(mx):
    c[i][0] = 1
    for j in range(1, i + 1):
        c[i][j] = c[i - 1][j - 1] + c[i - 1][j]


class Solution:
    def nthSmallest(self, n: int, k: int) -> int:
        ans = 0
        for i in range(49, -1, -1):
            if n > c[i][k]:
                n -= c[i][k]
                ans |= 1 << i
                k -= 1
                if k == 0:
                    break
        return ans

def _test():
    s = Solution()
    assert s.nthSmallest(4, 2) == 9
    assert s.nthSmallest(3, 1) == 4
    print("LC 3821 OK")


if __name__ == "__main__":
    _test()
