"""
LC 3632. Subarrays with XOR at Least K — Hard
https://leetcode.com/problems/subarrays-with-xor-at-least-k/
Best: 01-Trie count XOR >= k — O(n log A)
"""

from __future__ import annotations

from typing import List, Optional, Dict, Tuple, Set
import math
import heapq
import bisect
from collections import defaultdict, Counter, deque
from functools import lru_cache
from math import gcd, isqrt


class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        # Trie node: children[0/1], count
        trie = [[-1, -1, 0]]  # left, right, cnt

        def insert(x: int) -> None:
            node = 0
            trie[node][2] += 1
            for b in range(31, -1, -1):
                bit = (x >> b) & 1
                if trie[node][bit] == -1:
                    trie[node][bit] = len(trie)
                    trie.append([-1, -1, 0])
                node = trie[node][bit]
                trie[node][2] += 1

        def query(x: int, k: int) -> int:
            # count inserted y with x^y >= k
            node = 0
            res = 0
            for b in range(31, -1, -1):
                if node == -1:
                    break
                xb = (x >> b) & 1
                kb = (k >> b) & 1
                if kb == 0:
                    # branch that makes xor bit 1 all count
                    pref = 1 - xb
                    child = trie[node][pref]
                    if child != -1:
                        res += trie[child][2]
                    node = trie[node][xb]
                else:
                    # must take xor bit 1
                    node = trie[node][1 - xb]
            if node != -1:
                res += trie[node][2]
            return res

        ans = 0
        pre = 0
        insert(0)
        for v in nums:
            pre ^= v
            ans += query(pre, k)
            insert(pre)
        return ans


def _test():

    s = Solution()
    assert s.countSubarrays([3, 1, 2, 3], 2) == 6
    assert s.countSubarrays([0, 0, 0], 0) == 6

    print("LC 3632 OK")


if __name__ == "__main__":
    _test()
