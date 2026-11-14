"""
LC 818. Race Car — Hard
Best: BFS (pos, speed) — O(target log)
"""

from collections import deque


class Solution:
    def racecar(self, target: int) -> int:
        q = deque([(0, 1, 0)])  # pos, speed, steps
        seen = {(0, 1)}
        while q:
            pos, speed, steps = q.popleft()
            if pos == target:
                return steps
            # A
            np, ns = pos + speed, speed * 2
            if (np, ns) not in seen and 0 < np < 2 * target:
                seen.add((np, ns))
                q.append((np, ns, steps + 1))
            # R
            ns2 = -1 if speed > 0 else 1
            if (pos, ns2) not in seen:
                seen.add((pos, ns2))
                q.append((pos, ns2, steps + 1))
        return -1


def _test():
    s = Solution()
    assert s.racecar(3) == 2
    assert s.racecar(6) == 5
    print("LC 818 OK")


if __name__ == "__main__":
    _test()
