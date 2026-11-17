"""
LC 1601. Maximum Number of Achievable Transfer Requests — Hard
https://leetcode.com/problems/maximum-number-of-achievable-transfer-requests/
Best: Enumerate subsets of requests — O(2^m · n)
"""

from typing import List


class Solution:
    def maximumRequests(self, n: int, requests: List[List[int]]) -> int:
        m = len(requests)
        ans = 0
        for mask in range(1 << m):
            bal = [0] * n
            cnt = 0
            for i in range(m):
                if mask & (1 << i):
                    f, t = requests[i]
                    bal[f] -= 1
                    bal[t] += 1
                    cnt += 1
            if all(x == 0 for x in bal):
                ans = max(ans, cnt)
        return ans


def _test():
    s = Solution()
    assert s.maximumRequests(5, [[0, 1], [1, 0], [0, 1], [1, 2], [2, 0], [3, 4]]) == 5
    assert s.maximumRequests(3, [[0, 0], [1, 2], [2, 1]]) == 3
    assert s.maximumRequests(4, [[0, 3], [3, 1], [1, 2], [2, 0]]) == 4
    print("LC 1601 OK")


if __name__ == "__main__":
    _test()
