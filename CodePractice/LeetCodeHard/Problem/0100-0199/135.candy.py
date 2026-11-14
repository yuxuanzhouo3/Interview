"""
LC 135. Candy — Hard
Best: Two-pass ratings — O(n)
"""

from typing import List


class Solution:
    def candy(self, ratings: List[int]) -> int:
        n = len(ratings)
        candies = [1] * n
        for i in range(1, n):
            if ratings[i] > ratings[i - 1]:
                candies[i] = candies[i - 1] + 1
        for i in range(n - 2, -1, -1):
            if ratings[i] > ratings[i + 1]:
                candies[i] = max(candies[i], candies[i + 1] + 1)
        return sum(candies)


def _test():
    s = Solution()
    assert s.candy([1, 0, 2]) == 5
    assert s.candy([1, 2, 2]) == 4
    print("LC 135 OK")


if __name__ == "__main__":
    _test()
