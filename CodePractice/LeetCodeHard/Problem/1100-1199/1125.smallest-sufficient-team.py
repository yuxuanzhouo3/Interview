"""
LC 1125. Smallest Sufficient Team — Hard
https://leetcode.com/problems/smallest-sufficient-team/
Best: Bitmask DP people — O(n * 2^m)
"""

from typing import List


class Solution:
    def smallestSufficientTeam(self, req_skills: List[str], people: List[List[str]]) -> List[int]:
        m = len(req_skills)
        idx = {s: i for i, s in enumerate(req_skills)}
        n = len(people)
        pskill = [0] * n
        for i, skills in enumerate(people):
            mask = 0
            for s in skills:
                if s in idx:
                    mask |= 1 << idx[s]
            pskill[i] = mask

        N = 1 << m
        # dp[mask] = list of people indices
        dp = [None] * N
        dp[0] = []
        for mask in range(N):
            if dp[mask] is None:
                continue
            for i, sk in enumerate(pskill):
                nmask = mask | sk
                if nmask == mask:
                    continue
                cand = dp[mask] + [i]
                if dp[nmask] is None or len(cand) < len(dp[nmask]):
                    dp[nmask] = cand
        return dp[N - 1]


def _test():
    s = Solution()
    r = s.smallestSufficientTeam(
        ["java", "nodejs", "reactjs"],
        [["java"], ["nodejs"], ["nodejs", "reactjs"]],
    )
    assert sorted(r) == [0, 2]
    r2 = s.smallestSufficientTeam(
        ["algorithms", "math", "java", "reactjs", "csharp", "aws"],
        [
            ["algorithms", "math", "java"],
            ["algorithms", "math", "reactjs"],
            ["java", "csharp", "aws"],
            ["reactjs", "csharp"],
            ["csharp", "math"],
            ["aws", "java"],
        ],
    )
    assert len(r2) == 2
    print("LC 1125 OK")


if __name__ == "__main__":
    _test()
