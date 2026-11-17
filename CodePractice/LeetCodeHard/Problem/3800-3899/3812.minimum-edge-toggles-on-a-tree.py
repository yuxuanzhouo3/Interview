"""
LC 3812. Minimum Edge Toggles on a Tree — Hard
https://leetcode.com/problems/minimum-edge-toggles-on-a-tree/
Best: doocs/leetcode reference solution (verified on examples)
"""


class Solution:
    def minimumFlips(
        self, n: int, edges: list[list[int]], start: str, target: str
    ) -> list[int]:
        g = [[] for _ in range(n)]
        for i, (a, b) in enumerate(edges):
            g[a].append((b, i))
            g[b].append((a, i))

        ans = []

        def dfs(a: int, fa: int) -> bool:
            rev = start[a] != target[a]
            for b, i in g[a]:
                if b != fa and dfs(b, a):
                    ans.append(i)
                    rev = not rev
            return rev

        if dfs(0, -1):
            return [-1]
        ans.sort()
        return ans

def _test():
    s = Solution()
    assert s.minimumFlips(3, [[0,1],[1,2]], "010", "100") == [0]
    assert s.minimumFlips(7, [[0,1],[1,2],[2,3],[3,4],[3,5],[1,6]], "0011000", "0010001") == [1,2,5]
    assert s.minimumFlips(2, [[0,1]], "00", "01") == [-1]
    print("LC 3812 OK")


if __name__ == "__main__":
    _test()
