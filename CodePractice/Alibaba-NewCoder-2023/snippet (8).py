import collections
import sys

n = m = -1
A = []
for line in sys.stdin:
    a = line.split()
    if n == m == -1:
        n, m = int(a[0]) , int(a[1])
    elif not A:
        A =  list(map(int, a)) 
    else:
        print("ERROR")
    
# print(n, m , A)

ans = 0
freq = collections.defaultdict(int)
i = j = 0

while j < n: 
    freq[A[j]] += 1
    while freq[A[j]]>=m:
        freq[A[i]] -= 1
        i += 1 
    ans += i # [0, i-1] (i, j]; i don't need to reset to 0, we could make window; for [0,..., i-1] ... j] is good, so [0,..., i-1] ... j+k] is good, we only need to caluate [i, ... j+k] is good or not 
    j += 1

print(ans)
        

'''
1. array/query brute-force O(N**2); windows O(N)
for i in range(n):
    d = collections.defaultdict(int)
    d[A[i]] = 1
    for j in range(i+1, n):
        d[A[j]] += 1
        if d[A[j]] == m :
            ans += (n - j)
            # print(i, j, A[j])
            break
2. segment tree / query space: O(n + n-1 + n-2 + n-3 ... + 2 + 1) = O(n^2); time: O(NlogN)
                       [0, n-1]
                  [0, n-2]       [1,  n-1]
                        .....
               [0, 4]   [1, 5] 
         [0, 3]    [1,4]   [2, 5]
      [0, 2)  [1,3)    [2, 4]           [n-3, n-1]
    (mf1, vs1) (mf2, vs2)
   [0, 1) [1, 2)  [2, 3] [3, 4]        [n-3, n-2]  [n-2, n-3]
 
[0, 0] [1, 1]   ....   
(1,A[0])(1,A[1])
'''


