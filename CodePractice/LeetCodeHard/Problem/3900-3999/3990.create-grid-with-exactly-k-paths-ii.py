"""
LC 3990. Create Grid With Exactly K Paths II — Hard
https://leetcode.com/problems/create-grid-with-exactly-k-paths-ii/
Best: kamyu104 reference solution (verified on examples)
"""

# Time:  O((logk)^2)
# Space: O(1)

# constructive algorithms
class Solution(object):
    def createGrid(self, k):
        """
        :type k: int
        :rtype: List[str]
        """
        l = k.bit_length()
        m, n = 2*l, l+3
        result = [['#']*n for _ in range(m)]
        for i in range(l):
            r = 2*i
            result[r][i] = result[r][i+1] = result[r+1][i] = result[r+1][i+1] = '.'
            if not (k&(1<<i)):
                continue
            for c in range(i+2, n):
                result[r][c] = '.'
        for r in range(m):
            result[r][n-1] = '.'
        return ["".join(row) for row in result]


def _count_paths(grid):
    m, n = len(grid), len(grid[0])
    dp = [[0]*n for _ in range(m)]
    if grid[0][0] == '#':
        return 0
    dp[0][0] = 1
    for i in range(m):
        for j in range(n):
            if grid[i][j] == '#' or (i, j) == (0, 0):
                continue
            dp[i][j] = (dp[i-1][j] if i else 0) + (dp[i][j-1] if j else 0)
    return dp[-1][-1]


def _test():
    s = Solution()
    for k in (2, 3, 5, 7, 16):
        g = s.createGrid(k)
        assert _count_paths(g) == k
    print("LC 3990 OK")


if __name__ == "__main__":
    _test()
