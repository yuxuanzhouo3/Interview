import sys, bisect

def longest_sequence0(A, N): 
    # A 连续递增subarray -> longest_consecutive_increasing / longest_increased_subarray
    dp = [1] * N

    for i in range(1, N):
        if A[i-1] < A[i]:
           dp[i] = dp[i-1] + 1
        else:
           dp[i] = max(dp[i], dp[i-1])
    
    return max(dp)

    '''
    cur = mx = 1
    for i in range(1, N):
        if A[i-1] < A[i]:
           cur += 1
           mx = max(mx, cur)
        else:
           cur = 1
    return mx
    '''

def longest_sequence1(A, N):
    # A 非连续array最长递增 O(n*n) -> longest_sequence_increasing
    dp = [1] * N
    for i in range(N):
        for j in range(i):
            if A[j] < A[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

def longest_increasing_subsequence(arr):
    """
    求最长严格递增子序列的长度（非连续）
    使用贪心 + 二分查找，O(n log n)
    """
    tails = []
    for num in arr:
        pos = bisect.bisect_left(tails, num)  # 严格递增
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    return len(tails)


res = []
def func(N, x, y):
    '''
    z = list(zip(x, y))
    z = sorted(z, key=lambda z: (z[0], z[1]))
    Z = [z[0]]
    for i in range(1, len(z)):
        if z[i][0] == z[i-1][0]:
           N -= 1
           continue
        Z.append(z[i])
    Y =[ num[1] for num in Z ]
    return longest_increasing_subsequence(Y, N)
    '''

    # 1. 组合并排序：x 升序，x相同时 y 降序（关键！）
    items = list(zip(x, y))
    items.sort(key=lambda p: (p[0], -p[1]))
    
    # 2. 去重：x 相同的只保留一个（由于排序，保留的是 y 最大的那个）
    # 但实际上，由于 y 降序排列，相同的 x 不会形成递增，所以不需要显式去重
    # 直接提取 y 即可
    y_sorted = [p[1] for p in items]
    return longest_increasing_subsequence(y_sorted)
    
T = -1
for line in sys.stdin:          
    T = int(line.split()[0])
    break

N = -1
x, y = [], []
for line in sys.stdin:
    a = line.split()
    if N == -1: 
       N = int(a[0])
    elif len(x) == 0: 
        x = list(map(int, a))
    elif len(y) == 0:         
        y = list(map(int, a))
    else:
        res.append(func(N, x, y))
        N = int(a[0])
        x, y = [], []

   
res.append(func(N, x, y))
for r in res: 
    print(r)
