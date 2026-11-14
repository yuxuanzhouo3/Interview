"""
LC 753. Cracking the Safe — Hard
Best: Hierholzer Euler path on k-ary digraph — O(k^n)
"""

class Solution:
    def crackSafe(self, n: int, k: int) -> str:
        if n == 1:
            return "".join(str(i) for i in range(k))
        seen = set()
        ans = []
        start = "0" * (n - 1)

        def dfs(node):
            for x in map(str, range(k)):
                edge = node + x
                if edge not in seen:
                    seen.add(edge)
                    dfs(edge[1:])
                    ans.append(x)

        dfs(start)
        return "".join(ans) + start


def _test():
    s = Solution()
    r = s.crackSafe(1, 2)
    assert set(r) >= {"0", "1"}
    r2 = s.crackSafe(2, 2)
    assert len(r2) == 2**2 + 2 - 1
    print("LC 753 OK")


if __name__ == "__main__":
    _test()
