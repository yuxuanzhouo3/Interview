import sys, collections
from heapq import heappush

'''
for line in sys.stdin:
    a = line.split()
    print(int(a[0]) + int(a[1]))
    
1
8
2 1 1 2 2
2 1 1 1 4
2 1 1 2 2
2 2 2 1 4
0
0
1 1 1
1 1 1

3
'''

input = sys.stdin.readline
N = int(input())

for _ in range(N):
    M = int(input())
    A = list(map(int, input().split()))
    # print(A)
    hq = []
    d = collections.defaultdict(int)
    prev = set()
    for i in range(1, A[0] * 2 + 1, 2):
        d[(A[i], A[i+1])] += 1
        heappush(hq, (-d[(A[i], A[i+1])], A[i], A[i+1]) )
        prev.add((A[i], A[i+1]))

    for j in range(1, M):
        A = list(map(int, input().split()))
        cur = set()
        for i in range(1, A[0] * 2 + 1, 2):
            if (A[i], A[i+1]) not in prev: 
                d[(A[i], A[i+1])] = 1
            else:
                d[(A[i], A[i+1])] += 1
            cur.add((A[i], A[i+1]))
            heappush(hq, (-d[(A[i], A[i+1])], A[i], A[i+1]))
        
        prev = cur

print(-hq[0][0])
