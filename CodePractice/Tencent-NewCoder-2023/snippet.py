from pickle import EXT1
import sys
n = -1
s = []
for line in sys.stdin:
    a = line.split()
    if n == -1: n = int(a[0])
    elif not s:
        s = a[0]
    else:
        print("ERROR")
'''
# print(s)
ones = []
zeros = []
for i, ch in enumerate(s):
    if ch == "1" and not ones:
        ones.append(i)
    elif ch == "0" and not zeros:
        zeros.append(i)

for i in range(n-1, -1, -1):
    if s[i] == "1" and len(ones) == 1:
        ones.append(i)
    elif s[i] == "0" and len(zeros) == 1:
        zeros.append(i)
'''

def f(n):
    return n * (n + 1) / 2

# print("1 ==> ", ones, "0 ==> ", zeros)
n0, n1 = s.count("0"), s.count("1")
if n0 == n: 
    print(int(f(n0)))
    sys.exit()
if n1 == n:
    print(int(f(n1)))
    sys.exit()

s1, e1 = s.index("1"), s.rindex("1")  # n-1-s[::-1].index("1")
s0, e0 = s.index("0"), s.rindex("0")  # n-1-s[::-1].index("0")

ans = 0
if e1 <= s0 or e0 <= s1:
    ans = f(e1-s1+1) + f(e0-s0+1)
    # print(1, ans)
elif (s1 < s0 and e0 < e1):
    ans = max( f(n1), f(n0), f(s0) + f(n0) + f(e1-e0))
    # print(3, ans, f(n1), f(n0), f(s0) + f(n0) + f(e1-e0) )
elif (s0 < s1 and e1 < e0):
    ans = max( f(n1), f(n0), f(s1) + f(n1) + f(e0-e1))
    # print(4, ans, f(n1), f(n0), f(s0) + f(n0) + f(e1-e0) )
else:
    reminder0 = reminder1 = 0
    if e0 < e1:
        reminder1 = e1 - e0
        reminder0 = s1
    elif e1 < e0:
        reminder0 =  e0 - e1
        reminder1 = s0
    ans = max( f(e1-s1+1 - (n0-reminder0) ) + f(reminder0), f(e0-s0+1 - (n1-reminder1)) + f(reminder1))
    # print( 2, ans, e1-s1+1, (n0-reminder0), reminder0, e0-s0+1 , (n1-reminder1), reminder1 ) # 299953

print(int(ans))
