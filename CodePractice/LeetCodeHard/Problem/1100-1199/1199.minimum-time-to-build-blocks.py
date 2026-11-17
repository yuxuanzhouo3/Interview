"""
LC 1199. Minimum Time to Build Blocks — Hard
https://leetcode.com/problems/minimum-time-to-build-blocks/
Tags: Greedy, Heap
Best: 逆向 Huffman：优先合并最大两块并加 split
"""

from typing import List
import heapq


class Solution:
    def minBuildTime(self, blocks: List[int], split: int) -> int:
        # start with all workers; reverse thinking: combine two workers costs split
        h = blocks[:]
        heapq.heapify(h)
        while len(h) > 1:
            a = heapq.heappop(h)
            b = heapq.heappop(h)
            heapq.heappush(h, b + split)  # larger determines after split path
        return h[0]


def _test():
    sol = Solution()
    assert sol.minBuildTime([1], 1) == 1
    assert sol.minBuildTime([1, 2], 5) == 7
    assert sol.minBuildTime([1, 2, 3], 1) == 4
    print("LC 1199 OK")


if __name__ == "__main__":
    _test()

