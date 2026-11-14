"""
LC 499. The Maze III — Hard
https://leetcode.com/problems/the-maze-iii/
Best: Dijkstra on rolling stops — O(mn log(mn) * 4)
"""

from typing import List
import heapq


class Solution:
    def findShortestWay(
        self, maze: List[List[int]], ball: List[int], hole: List[int]
    ) -> str:
        m, n = len(maze), len(maze[0])
        dirs = [("d", 1, 0), ("l", 0, -1), ("r", 0, 1), ("u", -1, 0)]
        dist = {}
        heap = [(0, "", ball[0], ball[1])]  # dist, path, r, c
        dist[(ball[0], ball[1])] = (0, "")

        while heap:
            d, path, r, c = heapq.heappop(heap)
            if [r, c] == hole:
                return path
            if dist.get((r, c), (float("inf"),))[0] < d:
                continue
            if dist.get((r, c), (float("inf"), ""))[0] == d and dist[(r, c)][1] < path:
                continue
            for ch, dr, dc in dirs:
                nr, nc = r, c
                steps = 0
                while (
                    0 <= nr + dr < m
                    and 0 <= nc + dc < n
                    and maze[nr + dr][nc + dc] == 0
                ):
                    nr += dr
                    nc += dc
                    steps += 1
                    if [nr, nc] == hole:
                        break
                if steps == 0:
                    continue
                nd = d + steps
                np = path + ch
                prev = dist.get((nr, nc))
                if prev is None or nd < prev[0] or (nd == prev[0] and np < prev[1]):
                    dist[(nr, nc)] = (nd, np)
                    heapq.heappush(heap, (nd, np, nr, nc))
        return "impossible"


def _test():
    s = Solution()
    maze = [
        [0, 0, 0, 0, 0],
        [1, 1, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1],
        [0, 1, 0, 0, 0],
    ]
    assert s.findShortestWay(maze, [4, 3], [0, 1]) == "lul"
    maze2 = [
        [0, 0, 0, 0, 0],
        [1, 1, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1],
        [0, 1, 0, 0, 0],
    ]
    assert s.findShortestWay(maze2, [4, 3], [3, 0]) == "impossible"
    assert s.findShortestWay([[0, 0, 0]], [0, 0], [0, 2]) == "r"
    print("LC 499 OK")


if __name__ == "__main__":
    _test()
