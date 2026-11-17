"""
LC 1153. String Transforms Into Another String — Hard
https://leetcode.com/problems/string-transforms-into-another-string/
Tags: Hash Table, String, Graph
Best: 映射图：无环或存在未使用字符可破环
"""

from collections import defaultdict


class Solution:
    def canConvert(self, str1: str, str2: str) -> bool:
        if str1 == str2:
            return True
        mapping = {}
        for a, b in zip(str1, str2):
            if a in mapping and mapping[a] != b:
                return False
            mapping[a] = b
        # if str2 uses all 26 letters, no temp char to break cycles
        return len(set(str2)) < 26


def _test():
    sol = Solution()
    assert sol.canConvert("aabcc", "ccdee") is True
    assert sol.canConvert("leetcode", "codeleet") is False
    assert sol.canConvert("abcdefghijklmnopqrstuvwxyz", "bcdefghijklmnopqrstuvwxyza") is False
    print("LC 1153 OK")


if __name__ == "__main__":
    _test()

