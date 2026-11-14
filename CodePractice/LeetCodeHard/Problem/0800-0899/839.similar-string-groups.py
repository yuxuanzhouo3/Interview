"""
LC 839. Similar String Groups — Hard
Best: Union-Find similar pairs — O(n^2 * L)
"""

from typing import List


class Solution:
    def numSimilarGroups(self, strs: List[str]) -> int:
        n = len(strs)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def similar(a, b):
            diff = 0
            for x, y in zip(a, b):
                if x != y:
                    diff += 1
                    if diff > 2:
                        return False
            return True

        for i in range(n):
            for j in range(i + 1, n):
                if similar(strs[i], strs[j]):
                    parent[find(i)] = find(j)
        return sum(find(i) == i for i in range(n))


def _test():
    s = Solution()
    assert s.numSimilarGroups(["tars", "rats", "arts", "star"]) == 2
    print("LC 839 OK")


if __name__ == "__main__":
    _test()
