'''
Stack
│
├── Normal Stack
│
├── Monotonic Stack
│   ├── Increasing Stack
│   ├── Decreasing Stack
│   ├── Next Greater / Smaller
│   ├── Previous Greater / Smaller
│   └── Range min/max boundary
│
└── Expression Stack

'''

# ============================================================
# Monotonic Stack
# ============================================================

# 典型识别：
# 1. Next Greater / Next Smaller
# 2. Previous Greater / Previous Smaller
# 3. 区间内不能出现更大 / 更小元素
# 4. 某元素作为区间 min / max
# 5. 每个元素只需入栈、出栈一次 -> O(n)


# ------------------------------------------------------------
# Template 1: Monotonic Increasing Stack
# ------------------------------------------------------------

A = [1,4,2,5,7,1,3] # output 10
Monotonic_Increasing_Stack = []

for x in A:

    # pop elements greater than x
    while Monotonic_Increasing_Stack and Monotonic_Increasing_Stack[-1] > x:
        top = Monotonic_Increasing_Stack.pop()

    Monotonic_Increasing_Stack.append(x)

# stack:
# bottom -> top
# small  -> large
print(Monotonic_Increasing_Stack)

# ------------------------------------------------------------
# Template 2: Monotonic Decreasing Stack
# ------------------------------------------------------------

Monotonic_Decreasing_Stack = []

for x in A:

    while Monotonic_Decreasing_Stack and Monotonic_Decreasing_Stack[-1] < x:
        top = Monotonic_Decreasing_Stack.pop()

    Monotonic_Decreasing_Stack.append(x)

# stack:
# bottom -> top
# large  -> small

print(Monotonic_Decreasing_Stack)

# ============================================================
# Problem:
# endpoints are the two minimum elements of the interval
#
# Valid [l, r]:
#
# min(A[l+1:r]) >= max(A[l], A[r])
#
# intuition:
# "No smaller element can appear in the middle"
#       ->
# Monotonic Increasing Stack
# ============================================================

A = [1,4,2,5,7,1,3] # output 10
def count_valid_intervals(A):

    # [value, count]
    stack = []
    ans = 0

    for x in A:

        # x is smaller:
        # popped values can pair with x
        while stack and stack[-1][0] > x:
            ans += stack[-1][1]
            stack.pop()

        if not stack:
            stack.append([x, 1])

        elif stack[-1][0] == x:
            # all equal values can pair with x
            ans += stack[-1][1]

            # nearest smaller value can also pair with x
            if len(stack) > 1:
                ans += 1

            stack[-1][1] += 1

        else:
            # stack[-1] < x
            # only nearest smaller value can pair with x
            ans += 1

            stack.append([x, 1])

    return ans


# Example
A = [1, 4, 2, 5, 7, 1, 3]

print(count_valid_intervals(A))
# 10
