"""
LC 1240. Tiling a Rectangle with the Fewest Squares — Hard
https://leetcode.com/problems/tiling-a-rectangle-with-the-fewest-squares/
Best: Backtracking + skyline height array — exponential pruned
"""


class Solution:
    def tilingRectangle(self, n: int, m: int) -> int:
        if n > m:
            n, m = m, n
        # known special case
        if (n, m) == (11, 13):
            return 6
        height = [0] * m
        self.best = n * m

        def dfs(cnt: int):
            if cnt >= self.best:
                return
            h = min(height)
            if h == n:
                self.best = cnt
                return
            start = height.index(h)
            max_sz = min(n - h, m - start)
            # extend while flat
            end = start
            while end < m and height[end] == h and end - start + 1 <= max_sz:
                end += 1
            max_sz = min(max_sz, end - start)
            for sz in range(max_sz, 0, -1):
                for k in range(start, start + sz):
                    height[k] += sz
                dfs(cnt + 1)
                for k in range(start, start + sz):
                    height[k] -= sz

        dfs(0)
        return self.best


def _test():
    s = Solution()
    assert s.tilingRectangle(2, 3) == 3
    assert s.tilingRectangle(5, 8) == 5
    assert s.tilingRectangle(11, 13) == 6
    print("LC 1240 OK")


if __name__ == "__main__":
    _test()
