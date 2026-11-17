"""
LC 3311. Construct 2D Grid Matching Graph Layout — Hard
https://leetcode.com/problems/construct-2d-grid-matching-graph-layout/
Best: 方法一
"""

class Solution:
    def constructGridLayout(self, n: int, edges: list[list[int]]) -> list[list[int]]:
        g = [[] for _ in range(n)]
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)
        deg = [-1] * 5
        for x, ys in enumerate(g):
            deg[len(ys)] = x
        if deg[1] != -1:
            row = [deg[1]]
        elif deg[4] == -1:
            x = deg[2]
            for y in g[x]:
                if len(g[y]) == 2:
                    row = [x, y]
                    break
        else:
            x = deg[2]
            row = [x]
            pre = x
            x = g[x][0]
            while len(g[x]) > 2:
                row.append(x)
                for y in g[x]:
                    if y != pre and len(g[y]) < 4:
                        pre = x
                        x = y
                        break
            row.append(x)

        ans = [row]
        vis = [False] * n
        for _ in range(n // len(row) - 1):
            for x in row:
                vis[x] = True
            nxt = []
            for x in row:
                for y in g[x]:
                    if not vis[y]:
                        nxt.append(y)
                        break
            ans.append(nxt)
            row = nxt
        return ans

def _test():
    s = Solution()
    got = s.constructGridLayout(n=4, edges=[[0, 1], [0, 2], [1, 3], [2, 3]])
    assert got is not None and len(got)>0
    got = s.constructGridLayout(n=5, edges=[[0, 1], [1, 3], [2, 3], [2, 4]])
    assert got is not None and len(got)>0
    got = s.constructGridLayout(n=9, edges=[[0, 1], [0, 4], [0, 5], [1, 7], [2, 3], [2, 4], [2, 5], [3, 6], [4, 6], [4, 7], [6, 8], [7, 8]])
    assert got is not None and len(got)>0
    print("LC 3311 OK")

if __name__ == "__main__":
    _test()
