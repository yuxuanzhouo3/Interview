import sys,math,collections

input = sys.stdin.readline
N = int(input())
A = []
for _ in range(N):
    A.append( list(map(int, input().split())) )

'''''
def is_line_2d(ax, ay, bx, by, cx, cy):
    
    AB = (bx-ax, by-ay); 
    AC = (cx-ax, cy-ay); 

    Dot:
    AB · AC = |AB||AC| cosθ 
            = (bx-ax, by-ay) . (cx-ax, cy-ay) 
            = (bx-ax) * (cx-ax) + (by-ay) * ( cy-ay) 

    Cross:
    AB × AC = |AB||AC| sinθ
            = (bx-ax, by-ay) x (cx-ax, cy-ay) 
            = (bx-ax) * ( cy-ay) + (by-ay) * (cx-ax)

    Cross:
    P1 × (P2 x P3) x ... or 
    P1 × P2 + P2 x P3 + P1 × P3 = S(P1P2) + S(P1P3) + S(P2P3)

                Dot                    Cross

    公式       ax*bx + ay*by         ax*by - ay*bx

    几何意义    cosθ                  sinθ

    = 0        垂直                  平行/共线

    > 0        锐角                  逆时针/左侧  C 在 AB 的左侧（逆时针）

    < 0        钝角                  顺时针/右侧  C 在 AB 的右侧（顺时针）


    S(abc) = 1/2 × base × height

      = 1/2 × |AB| × |AC|sinθ

      = 1/2 × |AB × AC|

    S(abca) = 2 * S(abc) = |AB × AC|

    二维向量
    AB = (x, y)
     │
     ↕
    复数
    z = x + yi
     │
     ├── abs(z)
     │      → 向量长度
     │
     └── conj(a) * b
              │
              ├── real → Dot
              │            → cosθ / 垂直
              │
              └── imag → Cross
                           → sinθ
                           → 左转/右转
                           → 共线
                           → 面积

    
    # k12 = x1-x2 / y1-y2;  k23 = x2-x3 / y2-y3; 
    # k12 = |a| . |b| / |a| X |b| ; 
    return (bx-ax) * ( cy-ay) - (by-ay) * (cx-ax) == 0

def is_line_3d(ax, ay, az, bx, by, bz,  cx, cy, cz):
    
    Cross:
    AB × AC = |AB||AC| sinθ
            = (bx-ax, by-ay, bz-az) x (cx-ax, cy-ay, cz-az) 
            = (bx-ax) * ( cy-ay) - (by-ay) * (cx-ax) + (by-ay) * (cz-az) - (bz-az) * (cy-ay) + (bz-az) * (cx-ax) - (bx-ax) * (cz-az)

    Dot   → x*x + y*y + z*z → scalar → cosθ
    Cross → (y*z-z*y, z*x-x*z, x*y-y*x) → vector → sinθ
            sinθ = 0 ==> (0, 0, 0) 

    
    return (bx-ax) * ( cy-ay) - (by-ay) * (cx-ax) == 0 and  (by-ay) * (cz-az) - (bz-az) * (cy-ay) == 0 and (bz-az) * (cx-ax) - (bx-ax) * (cz-az) == 0
    

A.sort()
ans = 0
for i in range(N-2):
    ax, ay, az = A[i]
    for j in range(i+1, N-1):
        bx, by, bz = A[j]
        for k in range(j+1, N):
            cx, cy, cz = A[k]
            if is_line_3d(ax, ay, az, bx, by, bz,  cx, cy, cz):
                ans += 1

print(ans * 3)

'''
gcd = math.gcd
ans = 1

for i in range(N):

    ax, ay, az = A[i]

    cnt = {}
    same = 0
    best = 0

    for j in range(i + 1, N):

        bx, by, bz = A[j]

        dx = bx - ax
        dy = by - ay
        dz = bz - az

        # 重复点
        if dx == 0 and dy == 0 and dz == 0:
            same += 1
            continue

        g = gcd(gcd(abs(dx), abs(dy)), abs(dz))

        dx //= g
        dy //= g
        dz //= g

        # 统一方向
        if dx < 0 or \
           (dx == 0 and dy < 0) or \
           (dx == 0 and dy == 0 and dz < 0):
            dx = -dx
            dy = -dy
            dz = -dz

        key = (dx, dy, dz)

        c = cnt.get(key, 0) + 1
        cnt[key] = c

        if c > best:
            best = c

    ans = max(ans, best + same + 1)

print(ans)