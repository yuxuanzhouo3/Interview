"""
LC 4029. Elevator Requests IV — Hard
https://leetcode.com/problems/elevator-requests-iv/
Best: kamyu104 reference solution (verified on examples)
"""

from collections import defaultdict
# Time:  O(m^2), m = len(requests)
# Space: O(m)

import collections


# sort, dp
class Solution(object):
    def elevatorRequests(self, n, start, requests):
        """
        :type n: int
        :type start: int
        :type requests: List[List[int]]
        :rtype: int
        """
        INF = float("inf")
        lookup = collections.defaultdict(int)
        for a, f in requests:
            lookup[f] = max(lookup[f], a)
        pairs = sorted(lookup.items())
        if len(pairs) == 1:
            return max(abs(start-pairs[0][0]), pairs[0][1])
        dp = [[INF]*len(pairs) for _ in range(2)]
        dp[0][1] = max(abs(start-pairs[0][0]), pairs[0][1])
        dp[1][0] = max(abs(start-pairs[-1][0]), pairs[-1][1])
        result = INF
        for l in reversed(range(1, len(pairs))):
            new_dp = [[INF]*len(pairs) for _ in range(2)]
            for i in range(len(pairs)-l+1):
                j = i+l-1
                for s in range(2):
                    if dp[s][i] == INF:
                        continue
                    x = pairs[i-1][0] if s == 0 else pairs[j+1][0]
                    if l == 1:
                        result = min(result, max(dp[s][i]+abs(x-pairs[i][0]), pairs[i][1]))
                        continue
                    new_dp[0][i+1] = min(new_dp[0][i+1], max(dp[s][i]+abs(x-pairs[i][0]), pairs[i][1]))
                    new_dp[1][i] = min(new_dp[1][i], max(dp[s][i]+abs(x-pairs[j][0]), pairs[j][1]))
            dp = new_dp
        return result


# Time:  O(m^2 * logr), m = len(requests)
# Space: O(m)
import collections


# sort, binary search, dp

def _test():
    s = Solution()
    assert s.elevatorRequests(n=9, start=0, requests=[[0, 8], [6, 5]]) == 9
    assert s.elevatorRequests(n=8, start=5, requests=[[1, 7], [7, 3]]) == 7
    assert s.elevatorRequests(n=7, start=3, requests=[[0, 5], [0, 1], [6, 3]]) == 8
    print("LC 4029 OK")


if __name__ == "__main__":
    _test()
