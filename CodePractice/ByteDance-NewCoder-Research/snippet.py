import sys

'''
for line in sys.stdin:
    a = line.split()
    print(int(a[0]) + int(a[1]))
'''

input = sys.stdin.readline
N = int(input())
A = [ list(input().split()[0]) for _ in range(N) ]

# print(N, A)

res = [] * N

for j in range(N):
    i = 0
    L = len(A[j])
    S = A[j]
    ans = []
    # print(j, S, L)
    while i < L:
        if i + 2 < L and S[i] == S[i+1] == S[i+2]:
            while i + 2 < L and S[i] == S[i+1] == S[i+2]:
                i += 1
            ans += S[i]
            ans += S[i+1]
            '''
            if i + 3 < L and S[i] == S[i+1] and S[i+2] == S[i+3]:
                ans += S[i]
                ans += S[i+1]
                ans += S[i+2]
                i += 4
                continue
            else:
            '''
            i += 2
            continue
        else:
            ans += S[i]
            i += 1

    # print(ans)
    res.append("".join(ans))

# print(res)

final = []
for j in range(N):
    i = 0
    L = len(res[j])
    S = res[j]
    ans = []
    # print(j, S, L)
    while i < L:
        if i + 3 < L and S[i] == S[i+1] and S[i+2] == S[i+3]:
            ans += S[i]
            ans += S[i+1]
            ans += S[i+2]
            i += 4
            continue
        else:
            ans += S[i]
            i += 1

    # print(ans)
    final.append("".join(ans))

for f in final:
    print(f) 
    
    
    
