"""
LC 3486. Longest Special Path II — Hard
https://leetcode.com/problems/longest-special-path-ii/
Tags: Array, Hash Table, Tree, Depth-First Search, Prefix Sum
Best: 最优解
"""

class Solution:
  # Similar to 3425. Longest Special Path
  def longestSpecialPath(
      self,
      edges: list[list[int]],
      nums: list[int]
  ) -> list[int]:
    maxLength = 0
    minNodes = 1
    graph = [[] for _ in range(len(nums))]

    for u, v, w in edges:
      graph[u].append((v, w))
      graph[v].append((u, w))

    prefix = [0]
    lastSeenDepth = {}

    def dfs(
        u: int,
        prev: int,
        leftBoundary: list[int],
    ) -> None:
      nonlocal maxLength, minNodes
      prevDepth = lastSeenDepth.get(nums[u], 0)
      lastSeenDepth[nums[u]] = len(prefix)

      if prevDepth != 0:
        leftBoundary = sorted(leftBoundary + [prevDepth])[-2:]

      length = prefix[-1] - prefix[leftBoundary[0]]
      nodes = len(prefix) - leftBoundary[0]
      if length > maxLength or (length == maxLength and nodes < minNodes):
        maxLength = length
        minNodes = nodes

      for v, w in graph[u]:
        if v == prev:
          continue
        prefix.append(prefix[-1] + w)
        dfs(v, u, leftBoundary)
        prefix.pop()

      lastSeenDepth[nums[u]] = prevDepth

    dfs(0, -1, leftBoundary=[0, 0])
    return [maxLength, minNodes]

def _test():
    s = Solution()
    assert s.longestSpecialPath(edges=[[0,1,1],[1,2,3],[1,3,1],[2,4,6],[4,7,2],[3,5,2],[3,6,5],[6,8,3]], nums=[1,1,0,3,1,2,1,1,0]) == [9,3]
    assert s.longestSpecialPath(edges=[[1,0,3],[0,2,4],[0,3,5]], nums=[1,1,0,2]) == [5,2]
    print("LC 3486 OK")

if __name__ == "__main__":
    _test()
