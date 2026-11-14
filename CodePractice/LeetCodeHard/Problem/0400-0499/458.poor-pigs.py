"""
LC 458. Poor Pigs — Hard
Best: Information theory — (T+1)^pigs >= buckets
"""

import math


class Solution:
    def poorPigs(self, buckets: int, minutesToDie: int, minutesToTest: int) -> int:
        states = minutesToTest // minutesToDie + 1
        return math.ceil(math.log(buckets) / math.log(states)) if buckets > 1 else 0


def _test():
    s = Solution()
    assert s.poorPigs(1000, 15, 60) == 5
    assert s.poorPigs(4, 15, 15) == 2
    print("LC 458 OK")


if __name__ == "__main__":
    _test()
