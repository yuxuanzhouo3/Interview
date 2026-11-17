"""
LC 2030. Smallest K-Length Subsequence With Occurrences of a Letter — Hard
https://leetcode.com/problems/smallest-k-length-subsequence-with-occurrences-of-a-letter/
Best: monotonic stack + letter quota — O(n)
"""


class Solution:
    def smallestSubsequence(self, s: str, k: int, letter: str, repetition: int) -> str:
        stack = []
        required = repetition
        n_letters = s.count(letter)
        n = len(s)

        for i, c in enumerate(s):
            while (
                stack
                and stack[-1] > c
                and len(stack) + n - i - 1 >= k
                and (stack[-1] != letter or n_letters > required)
            ):
                if stack.pop() == letter:
                    required += 1
            if len(stack) < k:
                if c == letter:
                    stack.append(c)
                    required -= 1
                elif k - len(stack) > required:
                    stack.append(c)
            if c == letter:
                n_letters -= 1

        return "".join(stack)


def _test():
    s = Solution()
    assert s.smallestSubsequence("leet", 3, "e", 1) == "eet"
    assert s.smallestSubsequence("leetcode", 4, "e", 2) == "ecde"
    assert s.smallestSubsequence("bb", 2, "b", 2) == "bb"
    print("LC 2030 OK")


if __name__ == "__main__":
    _test()
