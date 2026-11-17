"""
LC 2876. Count Visited Nodes in a Directed Graph — Hard
https://leetcode.com/problems/count-visited-nodes-in-a-directed-graph/
Best: Find cycles then DP on chains — O(n)
"""

from typing import List


class Solution:
    def countVisitedNodes(self, edges: List[int]) -> List[int]:
        n = len(edges)
        ans = [0] * n
        state = [0] * n  # 0 not, 1 visiting, 2 done

        def dfs(u):
            if state[u] == 1:
                # found cycle start
                return
            if state[u] == 2:
                return
            state[u] = 1
            dfs(edges[u])
            state[u] = 2

        # better iterative approach
        for i in range(n):
            if ans[i]:
                continue
            path = []
            u = i
            seen = {}
            while u not in seen and ans[u] == 0:
                seen[u] = len(path)
                path.append(u)
                u = edges[u]
            if ans[u]:
                # hit computed
                dist = ans[u]
                for j in range(len(path) - 1, -1, -1):
                    dist += 1
                    ans[path[j]] = dist
            elif u in seen:
                # cycle
                cyc = path[seen[u] :]
                clen = len(cyc)
                for x in cyc:
                    ans[x] = clen
                dist = clen
                for j in range(seen[u] - 1, -1, -1):
                    dist += 1
                    ans[path[j]] = dist
        return ans


def _test():
    s = Solution()
    assert s.countVisitedNodes([1, 2, 0, 0]) == [3, 3, 3, 4]
    assert s.countVisitedNodes([1, 2, 3, 4, 0]) == [5, 5, 5, 5, 5]
    print("LC 2876 OK")


if __name__ == "__main__":
    _test()
