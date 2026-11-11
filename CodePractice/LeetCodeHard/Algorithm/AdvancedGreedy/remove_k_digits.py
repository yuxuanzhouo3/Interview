"""
5. 删 K 位使数字最小 — 单调栈贪心
App: LC 402。
"""


def remove_k_digits(num, k):
    st = []
    for ch in num:
        while k and st and st[-1] > ch:
            st.pop()
            k -= 1
        st.append(ch)
    while k:
        st.pop()
        k -= 1
    res = "".join(st).lstrip("0")
    return res or "0"


def _test():
    assert remove_k_digits("1432219", 3) == "1219"
    assert remove_k_digits("10200", 1) == "200"
    assert remove_k_digits("10", 2) == "0"
    print("remove_k_digits OK", remove_k_digits("1432219", 3))


APPLICATION = """
- 栈顶更大且还能删 → 弹掉，保证高位尽量小。
- 删完去前导零。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
