"""
LC 3753. Total Waviness of Numbers in Range II — Hard
https://leetcode.com/problems/total-waviness-of-numbers-in-range-ii/
Best: doocs/leetcode reference solution (verified on examples)
"""

from functools import cache
class Solution:
    def totalWaviness(self, num1: int, num2: int) -> int:
        def calc(x: int) -> int:
            if x < 0:
                return 0
            s = str(x)

            @cache
            def dfs(
                pos: int, prev2: int, prev1: int, started: int, limit: bool
            ) -> tuple:
                if pos == len(s):
                    return (started, 0)
                up = int(s[pos]) if limit else 9
                cnt = wav = 0
                for d in range(up + 1):
                    nlimit = limit and d == up
                    add = 0
                    if started == 0:
                        if d == 0:
                            ns, np2, np1 = 0, 10, 10
                        else:
                            ns, np2, np1 = 1, 10, d
                    else:
                        ns, np2, np1 = 1, prev1, d
                        if prev2 != 10 and (
                            (prev1 > prev2 and prev1 > d)
                            or (prev1 < prev2 and prev1 < d)
                        ):
                            add = 1
                    c, w = dfs(pos + 1, np2, np1, ns, nlimit)
                    cnt += c
                    wav += w + c * add
                return cnt, wav

            return dfs(0, 10, 10, 0, True)[1]

        return calc(num2) - calc(num1 - 1)

def _test():
    s = Solution()
    assert s.totalWaviness(120, 130) == 3
    assert s.totalWaviness(198, 202) == 3
    assert s.totalWaviness(4848, 4848) == 2
    print("LC 3753 OK")


if __name__ == "__main__":
    _test()
