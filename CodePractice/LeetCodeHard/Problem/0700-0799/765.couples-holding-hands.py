"""
LC 765. Couples Holding Hands — Hard
Best: Greedy swap / Union-Find couples — O(n)
"""

from typing import List


class Solution:
    def minSwapsCouples(self, row: List[int]) -> int:
        n = len(row)
        pos = {p: i for i, p in enumerate(row)}
        ans = 0
        for i in range(0, n, 2):
            a = row[i]
            b = a ^ 1
            if row[i + 1] == b:
                continue
            j = pos[b]
            # swap row[i+1] with row[j]
            partner = row[i + 1]
            row[i + 1], row[j] = row[j], row[i + 1]
            pos[partner] = j
            pos[b] = i + 1
            ans += 1
        return ans


def _test():
    s = Solution()
    assert s.minSwapsCouples([0, 2, 1, 3]) == 1
    assert s.minSwapsCouples([3, 2, 0, 1]) == 0
    print("LC 765 OK")


if __name__ == "__main__":
    _test()
