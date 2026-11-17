"""
LC 3544. Subtree Inversion Sum — Hard
https://leetcode.com/problems/subtree-inversion-sum/
Tags: Array, Dynamic Programming, Tree, Depth-First Search
Best: 最优解
"""
import functools

class Solution:
  def subtreeInversionSum(
      self,
      edges: list[list[int]],
      nums: list[int],
      k: int
  ) -> int:
    n = len(edges) + 1
    parent = [-1] * n
    graph = [[] for _ in range(n)]

    for u, v in edges:
      graph[u].append(v)
      graph[v].append(u)

    @functools.lru_cache(None)
    def dp(u: int, stepsSinceInversion: int, inverted: bool) -> int:
      """
      Returns the maximum sum for subtree rooted at u, with
      `stepsSinceInversion` steps of inversion and `inverted` is true if the
      subtree is inverted.
      """
      num = -nums[u] if inverted else nums[u]
      negNum = -num
      for v in graph[u]:
        if v == parent[u]:
          continue
        parent[v] = u
        num += dp(v, min(k, stepsSinceInversion + 1), inverted)
        if stepsSinceInversion == k:
          negNum += dp(v, 1, not inverted)
      return max(num, negNum) if stepsSinceInversion == k else num

    return dp(0, k, False)

def _test():
    s = Solution()
    assert s.subtreeInversionSum(edges=[[0,1],[0,2],[1,3],[1,4],[2,5],[2,6]], nums=[4,-8,-6,3,7,-2,5], k=2) == 27
    assert s.subtreeInversionSum(edges=[[0,1],[1,2],[2,3],[3,4]], nums=[-1,3,-2,4,-5], k=2) == 9
    assert s.subtreeInversionSum(edges=[[0,1],[0,2]], nums=[0,-1,-2], k=3) == 3
    print("LC 3544 OK")

if __name__ == "__main__":
    _test()
