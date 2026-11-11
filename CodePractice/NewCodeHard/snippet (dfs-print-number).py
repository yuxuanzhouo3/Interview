import sys

input = sys.stdin.readline

while True:
    n = input()
    if not n: break
    n = int(n)
    s = input().strip()
    res = []
    used = [0] * (n+1)

    def dfs(i, cur):
        if i >= len(s):
            if len(cur) == n: 
                res.append(map(str,cur[:]))
            return

        number = int(s[i])
        if 1<=number<10 and not used[number]: 
            used[number] = 1
            cur.append(number)
            dfs(i+1, cur)
            cur.remove(number)
            used[number] = 0

        if i < len(s) - 1:
            number = int(s[i:i+2])
            if 10<=number<=n and not used[number]: 
                used[number] = 1
                cur.append(number)
                dfs(i+2, cur)
                cur.remove(number)
                used[number] = 0
        
    
    dfs(0, [])
    print(' '.join(res[0]))