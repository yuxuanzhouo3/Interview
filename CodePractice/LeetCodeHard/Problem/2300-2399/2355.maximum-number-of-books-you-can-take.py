"""
LC 2355. Maximum Number of Books You Can Take — Hard
https://leetcode.com/problems/maximum-number-of-books-you-can-take/
Best: Monotonic stack + DP — O(n)
"""
from typing import List


class Solution:
    def maximumBooks(self, books: List[int]) -> int:
        n = len(books)
        dp = [0] * n
        stack = []
        for i, b in enumerate(books):
            while stack and books[stack[-1]] - stack[-1] >= b - i:
                stack.pop()
            j = stack[-1] if stack else -1
            cnt = min(b, i - j)
            dp[i] = (b + b - cnt + 1) * cnt // 2
            if j >= 0:
                dp[i] += dp[j]
            stack.append(i)
        return max(dp)


def _test():
    s = Solution()
    assert s.maximumBooks([8, 5, 2, 7, 9]) == 19
    assert s.maximumBooks([7, 0, 3, 4, 5]) == 12
    assert s.maximumBooks([8, 2, 3, 7, 3, 4, 0, 1, 4, 3]) == 13
    print("LC 2355 OK")


if __name__ == "__main__":
    _test()
