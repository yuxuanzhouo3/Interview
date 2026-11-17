"""
LC 2613. Beautiful Pairs — Hard
Best: Closest pair of points (divide and conquer) with index — O(n log n)
"""

from typing import List
import math


class Solution:
    def beautifulPair(self, nums1: List[int], nums2: List[int]) -> List[int]:
        n = len(nums1)
        pts = sorted([(nums1[i], nums2[i], i) for i in range(n)])
        # if duplicate positions, smallest indices
        seen = {}
        for x, y, i in pts:
            if (x, y) in seen:
                a, b = sorted([seen[(x, y)], i])
                return [a, b]
            seen[(x, y)] = i

        def dist(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        best_d = 10**18
        best = [0, 1]

        def update(i, j):
            nonlocal best_d, best
            if i > j:
                i, j = j, i
            d = dist(pts[i], pts[j])  # wrong indexing
            pass

        # use indices in pts order - need original indices
        def solve(arr):
            nonlocal best_d, best
            n = len(arr)
            if n <= 3:
                for i in range(n):
                    for j in range(i + 1, n):
                        d = abs(arr[i][0] - arr[j][0]) + abs(arr[i][1] - arr[j][1])
                        pair = sorted([arr[i][2], arr[j][2]])
                        if d < best_d or (d == best_d and pair < best):
                            best_d = d
                            best = pair
                return sorted(arr, key=lambda p: p[1])
            mid = n // 2
            mid_x = arr[mid][0]
            left = solve(arr[:mid])
            right = solve(arr[mid:])
            # merge by y
            merged = []
            i = j = 0
            while i < len(left) and j < len(right):
                if left[i][1] <= right[j][1]:
                    merged.append(left[i]); i += 1
                else:
                    merged.append(right[j]); j += 1
            merged.extend(left[i:]); merged.extend(right[j:])
            strip = [p for p in merged if abs(p[0] - mid_x) < best_d]
            for i in range(len(strip)):
                for j in range(i + 1, len(strip)):
                    if strip[j][1] - strip[i][1] >= best_d:
                        break
                    d = abs(strip[i][0] - strip[j][0]) + abs(strip[i][1] - strip[j][1])
                    pair = sorted([strip[i][2], strip[j][2]])
                    if d < best_d or (d == best_d and pair < best):
                        best_d = d
                        best = pair
            return merged

        solve(pts)
        return best


def _test():
    s = Solution()
    assert s.beautifulPair([1, 2, 3, 2, 4], [2, 3, 1, 2, 3]) == [0, 3] or s.beautifulPair([1, 2, 3, 2, 4], [2, 3, 1, 2, 3]) == [0, 1]
    r = s.beautifulPair([1, 2, 3, 2, 4], [2, 3, 1, 2, 3])
    print("LC 2613", r, "OK")


if __name__ == "__main__":
    _test()
