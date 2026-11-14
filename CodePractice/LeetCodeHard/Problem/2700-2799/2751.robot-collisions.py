"""
LC 2751. Robot Collisions — Hard
https://leetcode.com/problems/robot-collisions/
Best: Sort by position + stack for R vs L collisions — O(n log n)
"""

from typing import List


class Solution:
    def survivedRobotsHealths(self, positions: List[int], healths: List[int], directions: str) -> List[int]:
        n = len(positions)
        order = sorted(range(n), key=lambda i: positions[i])
        stack = []  # indices moving right surviving so far
        alive = [True] * n
        h = healths[:]
        for i in order:
            if directions[i] == "R":
                stack.append(i)
                continue
            # moving left: collide with stack rights
            while stack and alive[i]:
                j = stack[-1]
                if h[j] < h[i]:
                    alive[j] = False
                    h[i] -= 1
                    stack.pop()
                elif h[j] > h[i]:
                    alive[i] = False
                    h[j] -= 1
                else:
                    alive[i] = alive[j] = False
                    stack.pop()
        return [h[i] for i in range(n) if alive[i]]


def _test():
    s = Solution()
    assert s.survivedRobotsHealths([5, 4, 3, 2, 1], [2, 17, 9, 15, 10], "RRRRR") == [2, 17, 9, 15, 10]
    assert s.survivedRobotsHealths([3, 5, 2, 6], [10, 10, 15, 12], "RLRL") == [14]
    assert s.survivedRobotsHealths([1, 2, 5, 6], [10, 10, 11, 11], "RLRL") == []
    print("LC 2751 OK")


if __name__ == "__main__":
    _test()
