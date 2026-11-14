"""
LC 1439. Find the Kth Smallest Sum of a Matrix With Sorted Rows — Hard
https://leetcode.com/problems/find-the-kth-smallest-sum-of-a-matrix-with-sorted-rows/
Best: Merge row-by-row with heap — O(m k log k)
"""

from typing import List
import heapq


class Solution:
    def kthSmallest(self, mat: List[List[int]], k: int) -> int:
        def ksmall(a: List[int], b: List[int], k: int) -> List[int]:
            heap = [(a[0] + b[0], 0, 0)]
            seen = {(0, 0)}
            res = []
            while heap and len(res) < k:
                s, i, j = heapq.heappop(heap)
                res.append(s)
                if i + 1 < len(a) and (i + 1, j) not in seen:
                    seen.add((i + 1, j))
                    heapq.heappush(heap, (a[i + 1] + b[j], i + 1, j))
                if j + 1 < len(b) and (i, j + 1) not in seen:
                    seen.add((i, j + 1))
                    heapq.heappush(heap, (a[i] + b[j + 1], i, j + 1))
            return res

        cur = mat[0]
        for row in mat[1:]:
            cur = ksmall(cur, row, k)
        return cur[k - 1]


def _test():
    s = Solution()
    assert s.kthSmallest([[1, 3, 11], [2, 4, 6]], 5) == 7
    assert s.kthSmallest([[1, 3, 11], [2, 4, 6]], 9) == 17
    assert s.kthSmallest([[1, 10, 10], [1, 4, 5], [2, 3, 6]], 7) == 9
    print("LC 1439 OK")


if __name__ == "__main__":
    _test()
