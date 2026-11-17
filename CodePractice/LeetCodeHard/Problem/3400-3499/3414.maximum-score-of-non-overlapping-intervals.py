"""
LC 3414. Maximum Score of Non-overlapping Intervals — Hard
https://leetcode.com/problems/maximum-score-of-non-overlapping-intervals/
Best: 方法一：排序 + 二分查找 + 动态规划
"""

class Solution:
    def maximumWeight(self, intervals: list[list[int]]) -> list[int]:
        n = len(intervals)
        arr = [[e[0], e[1], e[2], i] for i, e in enumerate(intervals)]
        arr.sort()
        nxt = [0] * n
        for i in range(n):
            l, r = i + 1, n
            while l < r:
                mid = (l + r) >> 1
                if arr[mid][0] > arr[i][1]:
                    r = mid
                else:
                    l = mid + 1
            nxt[i] = l
        f = [[0] * 5 for _ in range(n + 1)]
        g = [[[] for _ in range(5)] for _ in range(n + 1)]
        for i in range(n - 1, -1, -1):
            for k in range(1, 5):
                s1, a1 = f[i + 1][k], g[i + 1][k]
                a2 = g[nxt[i]][k - 1][:]
                x = arr[i][3]
                j = 0
                while j < len(a2) and a2[j] < x:
                    j += 1
                a2.insert(j, x)
                s2 = f[nxt[i]][k - 1] + arr[i][2]
                if s2 > s1 or (s2 == s1 and a2 < a1):
                    f[i][k] = s2
                    g[i][k] = a2
                else:
                    f[i][k] = s1
                    g[i][k] = a1
        return g[0][4]

def _test():
    s = Solution()
    assert s.maximumWeight(intervals=[[1, 3, 2], [4, 5, 2], [1, 5, 5], [6, 9, 3], [6, 7, 1], [8, 9, 1]]) == [2, 3]
    assert s.maximumWeight(intervals=[[5, 8, 1], [6, 7, 7], [4, 7, 3], [9, 10, 6], [7, 8, 2], [11, 14, 3], [3, 5, 5]]) == [1, 3, 5, 6]
    print("LC 3414 OK")

if __name__ == "__main__":
    _test()
