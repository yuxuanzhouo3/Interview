from itertools import combinations
import sys, bisect

'''
for line in sys.stdin:
    a = line.split()
    print(int(a[0]) + int(a[1]))
'''

input = sys.stdin.readline
N, D = list(map(int, input().split()))
A = list(map(int, input().split()))

# print(N, D,  A)

M = 99997867
index = { a:i for i, a in enumerate(A)}
ans = 0

def a(num):
    number = 1
    for k in range(2, 1+num):
        number = (number  * k) 
    return number %M

def comb(L, K):
    # L! / ( K! * (L-K)! )
    if L == K: return 1
    number = L-K+1
    for k in range(L-K+2, L+1):
        number = (number  * k) 
    return number / a(K)

i = 0
while i < N:
    j = bisect.bisect_right(A, A[i] + D) - 1
    L = j - i
    cur = L * (L -1) // 2
    # print(i, j, L, a(L), a(3), cur)
    ans = (ans + cur ) %M
    i += 1
    '''
    # 这两个写法一起用，一般是错的。原因很简单：
    # C(L,3) 会一次统计当前窗口里所有三元组，但你把 i 跳到 j-1 后，会漏掉“跨窗口”的合法三元组。
    if L > 2:
        cur = comb(L, 3) %M
        # print(i, j, L, a(L), a(3), cur)
        ans = (ans + cur ) %M
        i = j - 1 
    else:
        i += 1
    '''

print(int(ans))