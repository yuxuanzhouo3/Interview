"""
LC 32. Longest Valid Parentheses — Hard
Best: Stack of indices — O(n)
"""

class Solution:
    def longestValidParentheses(self, s: str) -> int:
        st = [-1]
        ans = 0
        for i, c in enumerate(s):
            if c == "(":
                st.append(i)
            else:
                st.pop()
                if not st:
                    st.append(i)
                else:
                    ans = max(ans, i - st[-1])
        return ans


def _test():
    s = Solution()
    assert s.longestValidParentheses("(()") == 2
    assert s.longestValidParentheses(")()())") == 4
    assert s.longestValidParentheses("") == 0
    print("LC 32 OK")


if __name__ == "__main__":
    _test()
