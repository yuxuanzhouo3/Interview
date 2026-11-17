"""
LC 2193. Minimum Number of Moves to Make Palindrome — Hard
https://leetcode.com/problems/minimum-number-of-moves-to-make-palindrome/
Best: greedy pair from outside — O(n^2)
"""


class Solution:
    def minMovesToMakePalindrome(self, s: str) -> int:
        cs = list(s)
        ans, n = 0, len(s)
        i, j = 0, n - 1
        while i < j:
            found = False
            for k in range(j, i, -1):
                if cs[i] == cs[k]:
                    found = True
                    while k < j:
                        cs[k], cs[k + 1] = cs[k + 1], cs[k]
                        k += 1
                        ans += 1
                    j -= 1
                    break
            if not found:
                # unpaired middle char: cost to move to center
                ans += n // 2 - i
            i += 1
        return ans


def _test():
    s = Solution()
    assert s.minMovesToMakePalindrome("aabb") == 2
    assert s.minMovesToMakePalindrome("letelt") == 2
    print("LC 2193 OK")


if __name__ == "__main__":
    _test()
