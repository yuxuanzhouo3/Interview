"""
LC 1687. Delivering Boxes from Storage to Ports — Hard
https://leetcode.com/problems/delivering-boxes-from-storage-to-ports/
Best: DP + monoqueue — O(n)
"""

from collections import deque
from typing import List


class Solution:
    def boxDelivering(
        self, boxes: List[List[int]], portsCount: int, maxBoxes: int, maxWeight: int
    ) -> int:
        n = len(boxes)
        diff = [0] * n  # diff[i] = #port changes in boxes[0..i]
        for i in range(1, n):
            diff[i] = diff[i - 1] + (boxes[i][0] != boxes[i - 1][0])
        wpre = [0] * (n + 1)
        for i in range(n):
            wpre[i + 1] = wpre[i] + boxes[i][1]
        dp = [0] * (n + 1)
        q = deque([0])  # j = number of boxes already done

        def key(j):
            # dp[j] - diff[j]  (diff[-1] style: changes from j onward measured vs diff[i-1]-diff[j])
            return dp[j] - (diff[j] if j < n else 0)

        for i in range(1, n + 1):
            while q and (i - q[0] > maxBoxes or wpre[i] - wpre[q[0]] > maxWeight):
                q.popleft()
            j = q[0]
            changes = diff[i - 1] - (diff[j] if j < n else 0)
            # when j==0, diff[0]=0; changes in [0..i-1] = diff[i-1]
            # when j>0, changes in [j..i-1] = diff[i-1]-diff[j]
            # Wait: diff[j] includes changes up to index j. For segment boxes[j..i-1],
            # changes = diff[i-1] - diff[j]
            if j == 0:
                changes = diff[i - 1]
            else:
                changes = diff[i - 1] - diff[j]
            dp[i] = dp[j] + 2 + changes
            while q and key(q[-1]) >= key(i):
                q.pop()
            q.append(i)
        return dp[n]


def _test():
    s = Solution()
    assert s.boxDelivering([[1, 1], [2, 1], [1, 1]], 2, 3, 3) == 4
    assert s.boxDelivering([[1, 2], [3, 3], [3, 1], [3, 1], [2, 4]], 3, 3, 6) == 6
    assert s.boxDelivering([[1, 4], [1, 2], [2, 1], [2, 1], [3, 2], [3, 4]], 3, 6, 7) == 6
    print("LC 1687 OK")


if __name__ == "__main__":
    _test()
