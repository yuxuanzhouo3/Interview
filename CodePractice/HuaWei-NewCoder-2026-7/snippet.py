import sys, math

B = C = F = U = n = -1
A = []
for line in sys.stdin:
    a = line.split()
    if B == -1:
        B = int(a[0])
    elif C == -1:
        C = int(a[0])
    elif F == -1:
        F = int(a[0])
    elif U == -1:
        U = int(a[0])
    elif n == -1:
        n = int(a[0])
    elif not A:
        A = list(map(int,a))

# print("B=", B , "C=" , C ,  "F=" , F ,  "U=" ,  U ,  "n=" ,  n, A)

'''
ceiling(X / C) * F + X * U <= B
X * (F / C + U) <= B - F 
X <= (B-F) /  (F / C + U)
'''

# X = int( (B) /  (F / C + U) ) # X: max U could provide within the original B fees
A.sort()

# print(A, "B =" , (B) , "F / C =" , (F / C) , "F / C  +  U =" , (F / C) + U)

def fee(X): # 抬高需要 X 单位土的 费用
    return math.ceil(X / C) * F + X * U

cur = 0 # cur 单位土
f = n
ans = A[0]
for i in range(1, n):
    enhance = (A[i] - A[i-1]) * i
    if enhance == 0: continue
    X = cur + enhance
    # print("ans, cur, enhance, X, i, A[i] = ", (ans, cur, enhance, X, i, A[i]))
    if fee(X) <= B:
        cur += enhance
        ans += A[i] - A[i-1]
    else:
        f = i
        break

# print(ans, cur, n)

# if f == 0:
cur += f
while fee(cur) <= B:
    ans += 1
    cur += f

print(int(ans))
