"""
LC 679. 24 Game — Hard
https://leetcode.com/problems/24-game/
Best: backtracking over pairwise ops — O(1) (4 cards)
"""

from typing import List


class Solution:
    def judgePoint24(self, cards: List[int]) -> bool:
        EPS = 1e-6

        def dfs(nums):
            if len(nums) == 1:
                return abs(nums[0] - 24) < EPS
            for i in range(len(nums)):
                for j in range(len(nums)):
                    if i == j:
                        continue
                    nxt = [nums[k] for k in range(len(nums)) if k != i and k != j]
                    a, b = nums[i], nums[j]
                    cands = [a + b, a - b, a * b]
                    if abs(b) > EPS:
                        cands.append(a / b)
                    for v in cands:
                        if dfs(nxt + [v]):
                            return True
            return False

        return dfs([float(x) for x in cards])


def _test():
    s = Solution()
    assert s.judgePoint24([4, 1, 8, 3]) is True
    assert s.judgePoint24([1, 2, 3, 4]) is True
    assert s.judgePoint24([1, 1, 1, 1]) is False
    print("LC 679 OK")


if __name__ == "__main__":
    _test()
