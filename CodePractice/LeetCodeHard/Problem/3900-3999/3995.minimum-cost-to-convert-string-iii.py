"""
LC 3995. Minimum Cost to Convert String III — Hard
https://leetcode.com/problems/minimum-cost-to-convert-string-iii/
Best: kamyu104 reference solution (verified on examples)
"""

# Time:  O(n * r * l)
# Space: O(min(n, l))

# dp
class Solution(object):
    def minCost(self, source, target, rules, costs):
        """
        :type source: str
        :type target: str
        :type rules: List[List[str]]
        :type costs: List[int]
        :rtype: int
        """
        INF = float("inf")
        w = min(max(len(r) for _, r in rules), len(source)) + 1
        dp = [INF]*w
        dp[0] = 0
        for i in range(len(source)):
            dp[(i-1)%w] = INF
            if dp[i%w] is INF:
                continue
            if source[i] == target[i]:
                dp[(i+1)%w] = min(dp[(i+1)%w], dp[i%w])
            for j, (p, r) in enumerate(rules):
                c = costs[j]
                if i+len(p) >= len(source)+1:
                    continue
                for k in range(len(p)):
                    if r[k] != target[i+k]:
                        break
                    if p[k] == '*':
                        c += 1
                    elif p[k] != source[i+k]:
                        break
                else:
                    dp[(i+len(p))%w] = min(dp[(i+len(p))%w], dp[i%w]+c)
        return dp[len(source)%w] if dp[len(source)%w] is not INF else -1


# Time:  O(n * r * l)
# Space: O(n)
# dp

def _test():
    s = Solution()
    assert s.minCost(source='hello', target='world', rules=[['he', 'wo'], ['llo', 'rld']], costs=[3, 4]) == 7
    assert s.minCost(source='cat', target='dog', rules=[['c*t', 'dog']], costs=[2]) == 3
    assert s.minCost(source='test', target='next', rules=[['*e*t', 'next']], costs=[4]) == 6
    assert s.minCost(source='ab', target='bc', rules=[['a*', 'bd']], costs=[9]) == -1
    print("LC 3995 OK")


if __name__ == "__main__":
    _test()
