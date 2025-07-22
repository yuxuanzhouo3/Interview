'''
给定一个字符串 s 和一个整数 k，返回最长子串的长度，其中该子串中 每个字符至少出现 k 次。

python
复制
编辑
Input: s = "aaabb", k = 3
Output: 3
Explanation: "aaa" 是最长的有效子串。

Input: s = "ababbc", k = 2
Output: 5
Explanation: "ababb" 满足所有字符都至少出现两次。
'''
import collections


def LongestSubstringWithAtLeastKRepeatingCharacters(S, k):
    if not S: return 0
    if k<= 1: return len(S)
    # window, hashmap <-> bucket sort, heap, stack <-> recursion; dp; O(2*N) O(N) -> O(NlogN) O(N) | O(26N) O(N)
    N = len(S)
    res = 0

    for unique_target in range(1, 27): # Orginal We O(N^2) -> O(26*N*2) maintain a window contain at most unique_target chars
        l = r = 0
        unique = leastK = 0
        # d = collections.defaultdict(int)
        count = [0] * 26

        while r < N:
            ch = S[r]
            if count[ ord(ch) - ord('a')] == 0:
                unique += 1
            count[ord(ch) - ord('a')] += 1
            if count[ord(ch) - ord('a')] == k:
                leastK += 1
            r += 1 # [) -> res = r - l

            while unique > unique_target:
                ch = S[l]
                if count[ord(ch) - ord('a')] == k:
                    leastK -= 1
                count[ord(ch) - ord('a')] -= 1
                if count[ord(ch) - ord('a')] == 0:
                    unique -= 1
                l += 1

            if unique == leastK:
                res = max(res, r - l)

    return res