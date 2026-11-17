"""
LC 4023. Elevator Requests II — Hard
https://leetcode.com/problems/elevator-requests-ii/
Best: interval DP expanding covered floor segment left/right
"""


class Solution:
    def elevatorRequests(self, n: int, start: int, requests: list[int]) -> int:
        A = [x for x in requests if x != start]
        m = len(A)
        A.append(start)
        A.sort()
        s = A.index(start)
        sz = m + 1
        INF = 10**18
        dp = [[[INF, INF] for _ in range(sz)] for _ in range(sz)]
        dp[s][s][0] = dp[s][s][1] = 0
        for length in range(1, sz + 1):
            remaining = m - (length - 1)
            for l in range(sz - length + 1):
                r = l + length - 1
                if l > s or r < s:
                    continue
                if l > 0:
                    dp[l - 1][r][0] = min(
                        dp[l - 1][r][0],
                        dp[l][r][0] + (A[l] - A[l - 1]) * remaining,
                        dp[l][r][1] + (A[r] - A[l - 1]) * remaining,
                    )
                if r + 1 < sz:
                    dp[l][r + 1][1] = min(
                        dp[l][r + 1][1],
                        dp[l][r][0] + (A[r + 1] - A[l]) * remaining,
                        dp[l][r][1] + (A[r + 1] - A[r]) * remaining,
                    )
        return min(dp[0][sz - 1][0], dp[0][sz - 1][1])


def _test():
    s = Solution()
    assert s.elevatorRequests(6, 4, [1, 5]) == 6
    assert s.elevatorRequests(8, 3, [3, 7, 1]) == 10
    assert s.elevatorRequests(10, 5, [0, 2, 9]) == 22
    print("LC 4023 OK")


if __name__ == "__main__":
    _test()
