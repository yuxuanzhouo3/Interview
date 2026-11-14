"""
LC 927. Three Equal Parts — Hard
https://leetcode.com/problems/three-equal-parts/
Best: Count ones / align binary value — O(n)
"""

from typing import List


class Solution:
    def threeEqualParts(self, arr: List[int]) -> List[int]:
        ones = [i for i, v in enumerate(arr) if v == 1]
        total = len(ones)
        if total == 0:
            return [0, len(arr) - 1]
        if total % 3:
            return [-1, -1]
        t = total // 3
        i1, i2, i3 = ones[0], ones[t], ones[2 * t]
        # trailing zeros after third part
        trail = len(arr) - 1 - ones[-1]
        # ends of three parts (inclusive indices of last 1 plus trail)
        # part lengths of significant bits must match
        n = len(arr)
        # check patterns equal by comparing from each start
        def check(a, b, c):
            while c < n:
                if arr[a] != arr[b] or arr[b] != arr[c]:
                    return False
                a += 1
                b += 1
                c += 1
            return True

        # start of each part's first 1; first part may have leading zeros
        # ends: first ends at i1 + (pattern_len-1) + trail related
        # pattern from first 1 of each third to end of that third including trail zeros
        j = ones[t - 1] + trail
        k = ones[2 * t - 1] + trail
        # starts
        # third part must end at n-1; k must be end of second part before third
        if j >= ones[t] or k >= ones[2 * t]:
            return [-1, -1]
        # verify equality of three segments ending with same trail
        # segment1: [0..j], segment2: [j+1..k], segment3: [k+1..n-1]
        # compare from first 1
        p1, p2, p3 = ones[0], ones[t], ones[2 * t]
        while p3 < n:
            if arr[p1] != arr[p2] or arr[p2] != arr[p3]:
                return [-1, -1]
            p1 += 1
            p2 += 1
            p3 += 1
        if p1 != j + 1 or p2 != k + 1:
            return [-1, -1]
        return [j, k + 1]


def _test():
    s = Solution()
    assert s.threeEqualParts([1, 0, 1, 0, 1]) == [0, 3]
    assert s.threeEqualParts([1, 1, 0, 1, 1]) == [-1, -1]
    assert s.threeEqualParts([1, 1, 0, 0, 1]) == [0, 2]
    assert s.threeEqualParts([0, 0, 0, 0, 0]) == [0, 4]
    print("LC 927 OK")


if __name__ == "__main__":
    _test()
