import collections
import heapq


class Solution():
    def __init__(self):
        pass

    # 给定一个字符串 s，返回包含不同字符 (最多包含无限个不同字符，freq=1) 的最长子串的长度。
    def LongestSubstringWithoutRepeatingCharacters(self, S):
        if not S: return ""
        N = len(S)
        d = {} # d = set()
        i = res = 0
        for j in range(N):
            if S[j] in d and d[S[j]] >= i:
                i = d[S[j]] + 1
            else:
                res = max(res, j-i+1)
            d[S[j]] = j
        return res

    # 给定一个字符串 s，返回最多包含 两个不同字符 的最长子串的长度。
    def LongestSubstringwithAtMostTwoDistinctCharacters(self, S):
        if not S: return ""
        N = len(S) # O(1)
        d = collections.defaultdict(int) # set() set.remove(x) -> O(1)

        l = res = 0
        for r in range(N):
            d[S[r]] += 1
            while len(d) == 3 and l < r:
                d[S[l]] -= 1
                if d[S[l]] == 0:
                    del d[S[l]] # O(1)
                l += 1
            res = max(res, r - l + 1)
        return res

    # 给定一个字符串 s，返回最多包含 K个不同字符 的最长子串的长度。
    def LongestSubstringwithAtMostKDistinctCharacters(self, S, K):
        if not S or K < 1: return 0
        N = len(S) # O(1)
        d = collections.defaultdict(int) # set() set.remove(x) -> O(1)
        l = res = 0
        for r in range(N):
            d[S[r]] += 1
            while len(d) == K+1 and l < r:
                d[S[l]] -= 1
                if d[S[l]] == 0:
                    del d[S[l]] # O(1)
                l += 1
            res = max(res, r - l + 1)
        return res

    # 找出最长子串，可以通过最多K次替换使其所有字符都一样。
    # Input: S = "AABABBA", K = 1; Output: 4 ; 替换第一个 "B" 为 "A"，得到 "AAAABA"，最长连续相同字符长度为 4。
    def LongestRepeatingCharacterReplacement(self, S, K):
        if not S: return 0
        d = collections.defaultdict(int)  # set() set.remove(x) -> O(1)
        l = res = 0
        max_freq = 0 # window maxium freq char's freq
        for r in range(len(S)):
            d[S[r]] += 1
            max_freq = max(max_freq, d[S[r]])
            while (r-l+1) - max_freq > K and l < r:
                d[S[l]] -= 1
                if d[S[l]] == 0:
                    del d[S[l]]
                l += 1
            res = max(res, r - l + 1)
        return res


    def LongestSubstringWithAtLeastKRepeatingCharacters(S, k):
        if not S: return 0
        if k <= 1: return len(S)
        # window, hashmap <-> bucket sort, heap, stack <-> recursion; dp; O(2*N) O(N) -> O(NlogN) O(N) | O(26N) O(N)
        N = len(S)
        res = 0

        for unique_target in range(1, 27):  # Orginal We O(N^2) -> O(26*N*2) maintain a window contain at most unique_target chars
            l = r = 0
            unique = leastK = 0
            # d = collections.defaultdict(int)
            count = [0] * 26

            while r < N:
                ch = S[r]
                if count[ord(ch) - ord('a')] == 0:
                    unique += 1
                count[ord(ch) - ord('a')] += 1
                if count[ord(ch) - ord('a')] == k:
                    leastK += 1
                r += 1  # [) -> res = r - l

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


def minWindow(s, t):
    if not s or not t: return ""
    need = collections.Counter(t)
    start = 0
    min_len = float('inf')
    l = r = valid = 0
    window = collections.defaultdict(int)

    while r < len(s):
        c = s[r]
        r += 1
        window[c] += 1
        if window[c] == need[c]:
            valid += 1

        while valid == len(need):
            if r - l < min_len:
                start = l
                min_len = r - l

            d = s[l]
            if window[d] == need[d]:
                valid -= 1
            window[d] -= 1
            l += 1

    return s[start: start+min_len] if min_len!=float('inf') else ""



solv = Solution()
print('\nLongestRepeatingCharacterReplacement\n')








print('\nLongestRepeatingCharacterReplacement\n')

def test_LongestRepeatingCharacterReplacement():
    sol = Solution()
    cases = [
        # 🧪 边界情况
        ("", 0, 0),
        ("A", 0, 1),
        ("A", 1, 1),
        ("AA", 0, 2),
        ("AA", 1, 2),

        # ✅ 单字符替换
        ("AABABBA", 1, 4),
        ("ABAB", 2, 4),
        ("ABAB", 1, 3),

        # ✅ 全部相同
        ("AAAA", 2, 4),

        # ✅ 多字符混合
        ("ABCDE", 1, 2),
        ("ABCDE", 2, 3),

        # ✅ 替换跨多个字符
        ("ABAA", 0, 2),
        ("ABAA", 1, 4),
        ("ABAA", 2, 4),

        # ✅ 大小写敏感
        ("AaAaAaA", 1, 3),  # 'A' ≠ 'a'
        ("ABBB", 2, 4),

        # ✅ 替换量大于串长
        ("AABABBA", 100, 7),
        ("XYZXYZXYZ", 10, 9),
    ]

    for i, (s, k, expected) in enumerate(cases):
        result = sol.LongestRepeatingCharacterReplacement(s, k)
        status = "✅" if result == expected else "❌"
        print(f"Case {i}: Input=({s}, {k}) → Expected={expected}, Got={result} {status}")

test_LongestRepeatingCharacterReplacement()

print('\nLongestSubstringwithAtMostKDistinctCharacters\n')

def test_LongestSubstringwithAtMostKDistinctCharacters():
    sol = Solution()

    cases = [
        # 边界情况
        ("", 0, 0),  # 空串
        ("a", 0, 0),  # K=0
        ("a", 1, 1),  # 单字符 K=1

        # 普通情况
        ("eceba", 2, 3),  # "ece"
        ("aa", 1, 2),  # 全部一个字符
        ("aabbcc", 1, 2),  # 最长"a","b","c"
        ("aabbcc", 2, 4),  # "aabb", "bbcc"
        ("aabbcc", 3, 6),  # 全部

        # 高频变动 + 长度中等
        ("abcadcacacaca", 3, 11),  # "cadcacacaca"
        ("abcadcacacaca", 2, 8),  # "acacacaca" or "cacacaca"

        # 超过字符种类
        ("abcabcabc", 10, 9),  # 全部保留

        # 大小写不同
        ("aAaAaAaA", 1, 1),  # 'a', 'A'不同
        ("aAaAaAaA", 2, 8),  # 全部保留

        # Unicode 字符支持
        ("你好你好你好", 1, 1),  # 单一字符重复交错
        ("你好你好你好", 2, 6),  # 所有都保留
    ]

    for i, (s, k, expected) in enumerate(cases):
        result = sol.LongestSubstringwithAtMostKDistinctCharacters(s, k)
        print(f"Case {i}: Expected={expected}, Got={result}, {'✅' if result == expected else '❌'}")
# 调用测试
test_LongestSubstringwithAtMostKDistinctCharacters()
import random
import string
S = ''.join(random.choices(string.ascii_lowercase, k=10**5))
print(Solution().LongestSubstringwithAtMostKDistinctCharacters(S, 26))  # Expect 100000

print('\nLongestSubstringwithAtMostTwoDistinctCharacters\n')

print(solv.LongestSubstringwithAtMostTwoDistinctCharacters("eceba"))      # 3 → "ece"
print(solv.LongestSubstringwithAtMostTwoDistinctCharacters("ccaabbb"))    # 5 → "aabbb"
print(solv.LongestSubstringwithAtMostTwoDistinctCharacters("a"))          # 1
print(solv.LongestSubstringwithAtMostTwoDistinctCharacters("abcabcabc"))  # 2


print('\nLongestSubstringWithoutRepeatingCharacters\n')
print(solv.LongestSubstringWithoutRepeatingCharacters("abcabcbb"))  # 3
print(solv.LongestSubstringWithoutRepeatingCharacters("bbbbb")) # 1
print(solv.LongestSubstringWithoutRepeatingCharacters("pwwkew")) # 3