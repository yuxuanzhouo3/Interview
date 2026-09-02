import sys
from collections import deque

def solv(A, S, E, n, m):
    cnt_symetric = 0
    cnt_step = -1
    q = deque([S])
    d = [(-1, 0), (0, -1), (1, 0), (0, 1)]
    seen = set(q)
    #print(A, S, E, n, m , seen)
    while q:
        N = len(q)
        cnt_step += 1
        for _ in range(N):
            x, y, cnt_symetric = q.popleft()
            if (x, y) == E:
                return cnt_step

            for dx, dy in d:
                nx, ny = x+dx, y+dy
                if 0<=nx<n and 0<=ny<m and A[nx][ny] != '#' and (nx, ny, cnt_symetric) not in seen:
                    seen.add((nx, ny, cnt_symetric))
                    q.append((nx, ny, cnt_symetric))

            nx, ny = n-1-x, m-1-y # 中心对称
            if 0<=nx<n and 0<=ny<m and A[nx][ny]!= '#' and \
            (nx, ny, cnt_symetric+1) not in seen and cnt_symetric < 5:
                cnt_symetric += 1
                seen.add((nx, ny, cnt_symetric))
                q.append((nx, ny, cnt_symetric))
  
        #print(seen, q)
        
    return -1




n = m = -1
A = []
set_A = set()
for line in sys.stdin:
    line = line.strip()
    if not line : continue
    if m == -1:
        n, m = map(int, line.split())
        #print(m, n)
        continue
    if not (2<=n<=500 and 2<=m<=500): 
        sys.exit(1)
 
    a = list(line)
    #print(a)
    A.append(a)
    set_A |= set(a)

#print(A, set_A, set("#SE." ))
if set("#SE." ) != set_A: sys.exit(1)
S = E = []
for i in range(n):
    for j in range(m):
        if A[i][j] == 'S':
           if S: sys.exit(1)
           S = [i, j]
        if A[i][j] == 'E':
           if E: sys.exit(1)
           E = [i, j]

S += [0]
print(solv(A, tuple(S), tuple(E), n, m))



'''



'''