"""
LC 749. Contain Virus — Hard
https://leetcode.com/problems/contain-virus/
Best: Simulate regions + greedily wall most-threat — O((mn)^2)
"""

from typing import List, Set, Tuple


class Solution:
    def containVirus(self, isInfected: List[List[int]]) -> int:
        m, n = len(isInfected), len(isInfected[0])
        dirs = ((0, 1), (0, -1), (1, 0), (-1, 0))

        def neighbors(r, c):
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n:
                    yield nr, nc

        walls = 0
        while True:
            seen: Set[Tuple[int, int]] = set()
            regions = []  # (infected cells, threatened cells, wall_needed)
            for i in range(m):
                for j in range(n):
                    if isInfected[i][j] == 1 and (i, j) not in seen:
                        stack = [(i, j)]
                        seen.add((i, j))
                        cells = []
                        threat = set()
                        need = 0
                        while stack:
                            r, c = stack.pop()
                            cells.append((r, c))
                            for nr, nc in neighbors(r, c):
                                if isInfected[nr][nc] == 1 and (nr, nc) not in seen:
                                    seen.add((nr, nc))
                                    stack.append((nr, nc))
                                elif isInfected[nr][nc] == 0:
                                    threat.add((nr, nc))
                                    need += 1
                        regions.append((cells, threat, need))
            if not regions:
                break
            regions.sort(key=lambda x: len(x[1]), reverse=True)
            if not regions[0][1]:
                break
            # quarantine the most threatening region
            walls += regions[0][2]
            for r, c in regions[0][0]:
                isInfected[r][c] = -1
            # spread remaining
            to_infect = set()
            for cells, threat, _ in regions[1:]:
                to_infect |= threat
            for r, c in to_infect:
                isInfected[r][c] = 1
        return walls


def _test():
    s = Solution()
    g1 = [
        [0, 1, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
    assert s.containVirus([row[:] for row in g1]) == 10
    g2 = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    assert s.containVirus([row[:] for row in g2]) == 4
    g3 = [
        [1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 0, 0, 0],
    ]
    assert s.containVirus([row[:] for row in g3]) == 13
    print("LC 749 OK")


if __name__ == "__main__":
    _test()
