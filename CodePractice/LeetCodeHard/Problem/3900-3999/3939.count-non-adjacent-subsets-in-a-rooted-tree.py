"""
LC 3939. Count Non Adjacent Subsets in a Rooted Tree — Hard
https://leetcode.com/problems/count-non-adjacent-subsets-in-a-rooted-tree/
Best: tree knapsack DP (take / skip root) with modulo-k convolution
"""
MOD = 10**9 + 7


class Solution:
    def countValidSubsets(self, parent: list[int], nums: list[int], k: int) -> int:
        n = len(nums)
        adj = [[] for _ in range(n)]
        for i in range(1, n):
            adj[parent[i]].append(i)

        def dfs(u: int):
            yes = [0] * k
            no = [0] * k
            yes[nums[u] % k] = 1
            for v in adj[u]:
                vyes, vno = dfs(v)
                # merge into no: child may take or skip
                dpp = no[:]
                for i in range(k):
                    dpp[i] = (dpp[i] + vyes[i] + vno[i]) % MOD
                    if not no[i]:
                        continue
                    for j in range(k):
                        dpp[(j + i) % k] = (dpp[(j + i) % k] + no[i] * (vyes[j] + vno[j])) % MOD
                no = dpp
                # merge into yes: child must skip
                dpp = yes[:]
                for i in range(k):
                    if not yes[i]:
                        continue
                    for j in range(k):
                        dpp[(j + i) % k] = (dpp[(j + i) % k] + yes[i] * vno[j]) % MOD
                yes = dpp
            return yes, no

        a, b = dfs(0)
        return (a[0] + b[0]) % MOD


def _test():
    s = Solution()
    assert s.countValidSubsets([-1, 0, 1], [1, 2, 3], 3) == 1
    assert s.countValidSubsets([-1, 0, 0, 0], [2, 1, 2, 1], 3) == 2
    print("LC 3939 OK")


if __name__ == "__main__":
    _test()
