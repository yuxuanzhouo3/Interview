"""
LC 466. Count The Repetitions — Hard
https://leetcode.com/problems/count-the-repetitions/
Best: Cycle detection on s1 blocks — O(|s1| * |s2|)
"""


class Solution:
    def getMaxRepetitions(self, s1: str, n1: int, s2: str, n2: int) -> int:
        if n1 == 0:
            return 0
        # index_at_s1[k] = position in s2 after k copies of s1
        # count_at_s1[k] = how many s2 completed after k copies of s1
        recall = {}  # s2_idx -> (s1_count, s2_count)
        s1_count = s2_count = 0
        s2_idx = 0
        while s1_count < n1:
            s1_count += 1
            for c in s1:
                if c == s2[s2_idx]:
                    s2_idx += 1
                    if s2_idx == len(s2):
                        s2_count += 1
                        s2_idx = 0
            if s2_idx in recall:
                prev_s1, prev_s2 = recall[s2_idx]
                cycle_s1 = s1_count - prev_s1
                cycle_s2 = s2_count - prev_s2
                remain = n1 - s1_count
                cycles = remain // cycle_s1
                s1_count += cycles * cycle_s1
                s2_count += cycles * cycle_s2
            else:
                recall[s2_idx] = (s1_count, s2_count)
        return s2_count // n2


def _test():
    s = Solution()
    assert s.getMaxRepetitions("acb", 4, "ab", 2) == 2
    assert s.getMaxRepetitions("acb", 1, "acb", 1) == 1
    assert s.getMaxRepetitions("aaa", 3, "aa", 1) == 4
    assert s.getMaxRepetitions("abc", 0, "ab", 1) == 0
    print("LC 466 OK")


if __name__ == "__main__":
    _test()
