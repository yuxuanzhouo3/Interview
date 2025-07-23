def next_permutation(A):
    n = len(A)
    if n <= 1: return A

    i = n - 2

    # 1. 从右往左找到第一个下降点 i，满足 nums[i] < nums[i+1]
    while i >= 0 and A[i] >= A[i+1]:
        i -= 1

    # 如果找到了下降点 i
    if i >= 0:
        # 2. 从右往左找到第一个比 nums[i] 大的元素 j
        j = n - 1
        while j > i and A[j] <= A[i]:
            j -= 1

        # 3. 交换 nums[i] 和 nums[j]
        A[i], A[j] = A[j], A[i]

    # 4. 翻转 i+1 到末尾的元素，使后半部分升序
    l, r = i+1, n-1
    while l < r:
       A[l] , A[r] = A[r], A[l]
       l += 1
       r -= 1
    return A

# 测试
nums = [1, 3, 4, 2, 8, 7, 6, 5]
print("原始排列:", nums)
next_permutation(nums)
print("下一个排列:", nums)