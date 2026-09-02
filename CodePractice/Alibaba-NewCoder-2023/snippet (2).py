import sys
from functools import lru_cache

M = 10**9 + 7
@lru_cache(maxsize=None)
def solv(n, m):
    if n <=1: return 1 # 1 solution [] or [root]
    if 2**m-1 < n: return 0 # no solution [ error ]
    ans = 0
    for i in range(n):
        '''
        left = solv(i, m-1) % M
        right = solv(n-1-i, m-1) % M
        ans += (left if left else 1) * (right if right else 1)
        '''

        left = solv(i, m-1) % M
        right = solv(n-1-i, m-1) % M
        ans = (ans + left * right) % M
       
    return ans % M

'''
print(solv(3,2))
'''

for line in sys.stdin:
    a = line.split()
    print(solv(int(a[0]), int(a[1])))
    
