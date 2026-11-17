"""
LC 3816. Lexicographically Smallest String After Deleting Duplicate Characters — Hard
https://leetcode.com/problems/lexicographically-smallest-string-after-deleting-duplicate-characters/
Best: doocs/leetcode reference solution (verified on examples)
"""

from collections import defaultdict, Counter


class Solution:
    def lexSmallestAfterDeletion(self, s: str) -> str:
        cnt = Counter(s)
        stk = []
        for c in s:
            while stk and stk[-1] > c and cnt[stk[-1]] > 1:
                cnt[stk.pop()] -= 1
            stk.append(c)
        while stk and cnt[stk[-1]] > 1:
            cnt[stk.pop()] -= 1
        return "".join(stk)

def _test():
    s = Solution()
    assert s.lexSmallestAfterDeletion("aaccb") == "aacb"
    assert s.lexSmallestAfterDeletion("z") == "z"
    print("LC 3816 OK")


if __name__ == "__main__":
    _test()
