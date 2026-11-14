"""
LC 975. Odd Even Jump — Hard
https://leetcode.com/problems/odd-even-jump/
Best: SortedMap / monotonic from right — O(n log n)
"""

from typing import List
from sortedcontainers import SortedDict


class Solution:
    def oddEvenJumps(self, arr: List[int]) -> int:
        n = len(arr)
        odd_ok = [False] * n
        even_ok = [False] * n
        odd_ok[-1] = even_ok[-1] = True
        # next higher / next lower index
        # use sorted list of (val, idx) from right with mono stack approach via SortedDict
        try:
            from sortedcontainers import SortedDict as SD
        except ImportError:
            SD = None

        if SD is not None:
            sd = SD()
            higher = [None] * n
            lower = [None] * n
            for i in range(n - 1, -1, -1):
                v = arr[i]
                # ceil
                it = sd.bisect_left(v)
                keys = sd.keys()
                if it < len(keys):
                    higher[i] = sd[keys[it]]
                # floor
                it2 = sd.bisect_right(v) - 1
                if it2 >= 0:
                    lower[i] = sd[keys[it2]]
                sd[v] = i
            for i in range(n - 2, -1, -1):
                if higher[i] is not None:
                    odd_ok[i] = even_ok[higher[i]]
                if lower[i] is not None:
                    even_ok[i] = odd_ok[lower[i]]
            return sum(odd_ok)

        # fallback without sortedcontainers: sort indices
        odd_next = [None] * n
        even_next = [None] * n
        # mono stack for next greater-or-equal (smallest j>i with arr[j]>=arr[i] minimal)
        idx = sorted(range(n), key=lambda i: (arr[i], i))
        stack = []
        for i in idx:
            while stack and stack[-1] < i:
                odd_next[stack.pop()] = i
            stack.append(i)
        idx = sorted(range(n), key=lambda i: (-arr[i], i))
        stack = []
        for i in idx:
            while stack and stack[-1] < i:
                even_next[stack.pop()] = i
            stack.append(i)
        for i in range(n - 2, -1, -1):
            if odd_next[i] is not None:
                odd_ok[i] = even_ok[odd_next[i]]
            if even_next[i] is not None:
                even_ok[i] = odd_ok[even_next[i]]
        return sum(odd_ok)


def _test():
    s = Solution()
    assert s.oddEvenJumps([10, 13, 12, 14, 15]) == 2
    assert s.oddEvenJumps([2, 3, 1, 1, 4]) == 3
    assert s.oddEvenJumps([5, 1, 3, 4, 2]) == 3
    print("LC 975 OK")


if __name__ == "__main__":
    _test()
