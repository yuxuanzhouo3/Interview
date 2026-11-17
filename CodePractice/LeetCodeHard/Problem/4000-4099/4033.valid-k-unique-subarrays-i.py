"""
LC 4033. Valid K Unique Subarrays I — Hard
https://leetcode.com/problems/valid-k-unique-subarrays-i/
Best: random XOR hash for even freq + distinct count via Mo or offline
"""
import random
from collections import defaultdict


class Solution:
    def validSubarrays(self, nums: list[int], k: int, queries: list[list[int]]) -> list[bool]:
        # For samples n small enough — compute directly; also provide hash approach.
        # Even frequencies <=> xor of per-occurrence random masks is 0 if each value appears even times
        # Distinct count: need exactly k distinct — use prefix Mo or per-query scan for correctness on tests.
        n = len(nums)
        # Assign two random 64-bit values per number for stronger even-check (xor pair)
        rnd = {}
        for x in set(nums):
            rnd[x] = random.getrandbits(64)
        pxor = [0] * (n + 1)
        for i, x in enumerate(nums):
            pxor[i + 1] = pxor[i] ^ rnd[x]

        ans = []
        for l, r in queries:
            # even freqs
            if pxor[r + 1] ^ pxor[l] != 0:
                ans.append(False)
                continue
            # distinct + even check explicitly (xor can false-positive rarely; verify)
            freq = defaultdict(int)
            for i in range(l, r + 1):
                freq[nums[i]] += 1
            if len(freq) != k:
                ans.append(False)
                continue
            ans.append(all(v % 2 == 0 for v in freq.values()))
        return ans


def _test():
    s = Solution()
    assert s.validSubarrays([1, 2, 2, 1], 2, [[0, 1], [0, 3], [1, 2]]) == [False, True, False]
    assert s.validSubarrays([3, 3, 3], 1, [[1, 2], [0, 2]]) == [True, False]
    print("LC 4033 OK")


if __name__ == "__main__":
    _test()
