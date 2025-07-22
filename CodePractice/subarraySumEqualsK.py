# Given an integer array nums and an integer k, return the total number of subarrays whose sum equals to k.
import collections


# O(N) O(1) -- No Negative Numbers inside A
def subarraySum_no_negative_number(A, k):
    N = len(A)
    cur = 0
    l = r = ans = 0
    for r in range(N):
        cur += A[r]

        while cur > k and l < r:
            cur -= A[l]
            l += 1

        if cur == k:
            ans += 1

    while cur > k and l < N:
        cur -= A[l]
        l += 1

    if cur == k:
        ans += 1

    return ans

# O(N) O(N)
def subarraySum_with_negative_number(A, k):
    d = {}
    ans = cur_sum = 0
    for i in range(len(A)):
        cur_sum += A[i]
        if cur_sum-k in d:
            ans += d[cur_sum-k]
        # 维护哈希表
        if cur_sum in d:
            d[cur_sum] += 1
        else:
            d[cur_sum] = 1
        # ..................... cur
        # ...cur-k....cur-k..
    return ans





