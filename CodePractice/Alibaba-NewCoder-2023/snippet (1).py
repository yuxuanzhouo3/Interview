import sys

M = 10 ** 9 + 7

def cal(n1, n2, n):
    """
    已知 a+b = n1, ab = n2
    求 a^n + b^n mod M
    适用于所有 n（包括偶数）
    """
    
    # 边界条件
    if n == 0:
        return 2 % M
    if n == 1:
        return n1 % M
    
    # 递推：S_i = n1 * S_{i-1} - n2 * S_{i-2}
    s_prev2 = 2 % M          # S_0
    s_prev1 = n1 % M         # S_1
    
    for i in range(2, n + 1):
        s_curr = (n1 * s_prev1 - n2 * s_prev2) % M
        s_prev2, s_prev1 = s_prev1, s_curr
    
    return s_prev1

# 读取输入
N = -1
ans = []

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    
    if N == -1:
        N = int(line)
    else:
        n1, n2, n = map(int, line.split())
        ans.append(str(cal(n1, n2, n)))

# 输出结果
print("\n".join(ans))