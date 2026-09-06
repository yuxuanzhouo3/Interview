import sys

N = -1
A = []
for line in sys.stdin:
    a = line.split()
    if N == -1:
        N = int(a[0])
    elif not A:
        A = list(map(int, a))

# print(N, A)
if N <= 1: print(0)

inc_stack = []
ans = 0

for i, num in enumerate(A):
    while inc_stack and inc_stack[-1][0] > num:
        ans += inc_stack[-1][1]
        inc_stack.pop(-1)
    
    
    if not inc_stack:
        inc_stack.append([num, 1])

    elif inc_stack[-1][0] == num:
        ans += inc_stack[-1][1]

        if len(inc_stack) > 1:
            ans += 1

        inc_stack[-1][1] += 1

    else:
        # inc_stack[-1][0] < num
        ans += 1
        inc_stack.append([num, 1])


print(ans)


'''
O(N) 
rdp = [0] * N 
ldp = [0] * N

mn = A[0]
mx = A[-1]

for i in range(1, N):
    ldp[i] = A[i]

rdp[i] -> right numbers smaller than current i; 
ldp[i] -> left  numbers smmaler than current i;

a1, a2, a3, ... ai, ak, aj, ...,  an

ak -> left -> ai < ai+1 + LCS dp incresed
ak -> right -> aj > ak + LCS dp decresed

'''

'''
O(N**2)

ans = N - 1
best = (10**9+1, -1)
second = (10**9+2, -1, 0)
dp = [(best, second)] * N
cur = 10**9+1

for i in range(N):
    if A[i] <= cur:
        cur = A[i]
        if cur <= best[0]:
            x, y = best
            if best[0] == second[0]:
                f = second[2] + 1
                second = (x, y, f)
            else:
                second = (x, y, 0)
            best = (cur, i)
            cur = second[0]
        elif cur <= second[0]:
            f = second[2]
            f += (cur == second[0])
            second = (cur, i, f)
        else:
            cur = second[0]
    # print(i, best, second)
    dp[i] = (best, second)

print(dp)

for i in range(2, N):
    best, second = dp[i]
    v1, idx1 = best
    v2, idx2, f = second
    if idx1 == i:
        ans += 1 + f
    elif idx2 == i:
        ans += 1

print(ans)
'''   

