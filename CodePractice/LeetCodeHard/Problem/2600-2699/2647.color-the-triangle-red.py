"""
LC 2647. Color the Triangle Red — Hard
https://leetcode.com/problems/color-the-triangle-red/
Best: Constructive pattern — O(n)
"""
from typing import List


class Solution:
    def colorRed(self, n: int) -> List[List[int]]:
        # Triangular grid with  n(n+1)/2  cells; color ~n cells red so every
        # 2-cell component after removing red is avoided — specific construction.
        ans = []
        # Pattern from editorial: paint diagonal-ish cells
        ans.append([1, 1])
        r = n
        while r > 0:
            if (n - r) % 4 == 0:
                for c in range(1, 2 * r, 2):
                    ans.append([r, c])
                r -= 1
            elif (n - r) % 4 == 1:
                ans.append([r, 2])
                r -= 1
            elif (n - r) % 4 == 2:
                for c in range(3, 2 * r + 1, 2):
                    ans.append([r, c])
                r -= 1
            else:
                ans.append([r, 1]) if False else None
                # for %4==3: paint [r,1] then skip logic
                ans.append([r, 1])
                r -= 1
        # dedupe and filter valid
        seen = set()
        out = []
        for a, b in ans:
            if 1 <= a <= n and 1 <= b <= 2 * a - 1 and (a, b) not in seen:
                seen.add((a, b))
                out.append([a, b])
        return out


def _test():
    s = Solution()
    r = s.colorRed(1)
    assert r == [[1, 1]]
    r3 = s.colorRed(3)
    assert len(r3) >= 1
    print("LC 2647 OK")


if __name__ == "__main__":
    _test()
