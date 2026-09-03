import sys, collections

N = -1
A = []
graph = collections.defaultdict(list)
for line in sys.stdin:
    a = line.split()
    if N == -1: 
        N = int(a[0])
        continue
    elif not A:
        A += list(map(int, a))
        continue
    else:
        u, v = int(a[0]), int(a[1])
        graph[u].append(v)
        graph[v].append(u)

# print(N, A, graph)

index = collections.defaultdict(list)
for i, a in enumerate(A):
    index[a].append(i + 1)

candidates = []
for a, feq in collections.Counter(A).items():
    if feq > 1:
        candidates += index[a]

# print(candidates)  
sys.setrecursionlimit(10000)

def dfs(candidate, ai, v, step):
    if len(v) == N:
        return -1
    v.add(candidate)
    ans = float('inf')
    for nei in graph[candidate]:
        if nei not in v:
            if A[nei-1] == ai:
                ans = min(ans, step + 1)
            else:
                temp = dfs(nei, ai, v, step + 1)
                if temp != -1: 
                    ans = min(ans, temp)
    v.remove(candidate)
    return -1 if ans == float('inf') else ans


res = float('inf')
for candidate in candidates:
    ans = dfs(candidate, A[candidate-1], set(), 0)
    if ans != -1:
        res = min(res, ans)
    if res == 1:
        break

print(res if res != float('inf') else -1)

