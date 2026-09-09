import sys

B = n = -1
A = []
k = 0

for line in sys.stdin:
    a = line.split()
    if B == -1:
        B = int(a[0])
    elif n == -1:
        n = int(a[0])
    elif k < n:
        k += 1
        A.append(list(map(int, a)))
    
# print(B, n, A)

# A.sort(key=lambda x: -x[3])
# find all combs which is smaller than B, and later find the max satisfy
'''
C = []
for L in range(1, n+1):
    if A[L-1][2] <= B:
        C.append([L-1])
    else:
        if A[L-1][2] >= 200 and A[L-1][2] - 20 <= B:
            C.append([L-1])
'''

ans = [0]
g = [0] * 11
mask = 0

def dfs2(i, t, mask, S):
    if i == n: 
        p1 = t - (t // 200) * 20
        p2 = t - (bin(mask).count("1") >= 3) * (t // 200) * 30
        if min(p1, p2) <= B:
            ans[0] = max(ans[0], S)
        return

    _, gid, p, s = A[i]
      
    # mask |= (1 << (gid-1)) 
    dfs2(i+1, 
    t+p , 
    mask | (1 << (gid-1)) , 
    S+s)
    # mask |= (1 >> (gid-1)) 
    
    dfs2(i+1, 
        t, 
        mask, 
        S
        )


def dfs(i, t, g, S):
    if i == n: 
        p1 = t - (t // 200) * 20
        p2 = t - (sum(g) >= 3) * (t // 200) * 30
        if min(p1, p2) <= B:
            ans[0] = max(ans[0], S)
        return

    _, gid, p, s = A[i]
    
    g[gid] += 1
    dfs(i+1, t+p , g, S+s)
    g[gid] -= 1

    dfs(i+1, t, g, S)

# dfs(0, 0, g, 0)  # 0.6s, 4800kb
dfs2(0, 0, 0, 0) # 0.6s, 4772kb
print(ans[0])


