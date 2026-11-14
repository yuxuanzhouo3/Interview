"""
LC 60. Permutation Sequence — Hard
Best: Factorial number system — O(n^2)
"""

class Solution:
    def getPermutation(self, n: int, k: int) -> str:
        fact = [1] * n
        for i in range(1, n):
            fact[i] = fact[i - 1] * i
        nums = list(range(1, n + 1))
        k -= 1
        ans = []
        for i in range(n - 1, -1, -1):
            idx = k // fact[i]
            k %= fact[i]
            ans.append(str(nums.pop(idx)))
        return "".join(ans)


def _test():
    s = Solution()
    assert s.getPermutation(3, 3) == "213"
    assert s.getPermutation(4, 9) == "2314"
    print("LC 60 OK")


if __name__ == "__main__":
    _test()
