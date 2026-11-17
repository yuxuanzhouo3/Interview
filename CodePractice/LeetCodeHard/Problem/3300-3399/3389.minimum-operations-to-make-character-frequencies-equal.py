"""
LC 3389. Minimum Operations to Make Character Frequencies Equal — Hard
https://leetcode.com/problems/minimum-operations-to-make-character-frequencies-equal/
Best: Enumerate target freq + DP over alphabet — O(26 n)
"""


class Solution:
    def makeStringGood(self, s: str) -> int:
        count = [0] * 26
        for c in s:
            count[ord(c) - 97] += 1
        return min(self._cost(count, t) for t in range(1, max(count) + 1))

    def _cost(self, count, target: int) -> int:
        dp = [0] * 27
        for i in range(25, -1, -1):
            delete_all = count[i]
            to_target = abs(target - count[i])
            dp[i] = min(delete_all, to_target) + dp[i + 1]
            if i + 1 < 26 and count[i + 1] < target:
                deficit = target - count[i + 1]
                need = count[i] if count[i] <= target else count[i] - target
                change = need + deficit - need if deficit > need else deficit + need - deficit
                # simplify: abs(need - deficit) + min(need, deficit) == need + deficit - 2*min + min? 
                # walkccc: need+deficit-need = deficit if deficit>need else deficit+(need-deficit)=need
                change = max(need, deficit)  # equivalent: transfer min then fix rest
                dp[i] = min(dp[i], change + dp[i + 2])
        return dp[0]


def _test():
    s = Solution()
    assert s.makeStringGood("acab") == 1
    assert s.makeStringGood("wddw") == 0
    assert s.makeStringGood("aaabc") == 2
    print("LC 3389 OK")


if __name__ == "__main__":
    _test()
