"""
LC 1354. Construct Target Array With Multiple Sums — Hard
https://leetcode.com/problems/construct-target-array-with-multiple-sums/
Best: Reverse with max-heap — O(n log n · log S)
"""

import heapq
from typing import List


class Solution:
    def isPossible(self, target: List[int]) -> bool:
        if len(target) == 1:
            return target[0] == 1
        total = sum(target)
        heap = [-x for x in target]
        heapq.heapify(heap)
        while True:
            mx = -heapq.heappop(heap)
            if mx == 1:
                return True
            rest = total - mx
            if rest == 0 or mx <= rest:
                return False
            # replace mx with mx % rest (jump multiple steps)
            nxt = mx % rest
            if nxt == 0:
                nxt = rest  # need at least become rest? actually if mx%rest==0, previous would be rest, only ok if rest==1
                # if nxt==0, the value before ops would be multiple; only valid if we land on rest when rest==1? 
                # Standard: nxt = mx % rest; if nxt==0 return rest==1 somehow
                return rest == 1
            total = total - mx + nxt
            heapq.heappush(heap, -nxt)


def _test():
    s = Solution()
    assert s.isPossible([9, 3, 5]) is True
    assert s.isPossible([1, 1, 1, 2]) is False
    assert s.isPossible([8, 5]) is True
    assert s.isPossible([1, 1000000000]) is True
    print("LC 1354 OK")


if __name__ == "__main__":
    _test()
