import sys, math

n = m = -1
for line in sys.stdin:
    a = line.split()
    if n == m == -1:
        n, m = int(a[0]), int(a[1])
        break

# print(n, m)
exp = sum( 2 + n / i for i in range(1, 1+m) )
print(f"{exp:.2f}")

'''

k numbers of ssr left
p =  m - k / m + n - k  = k / n + k  ssr
q =  n / m + n - k = n / n + k       nor >= 0

Sk = 1 + x + x**2 + x**3 + .... x**k
xSk =    x + x**2 + x**3 + .... x**k + x**(k+1)
(1-x)Sk = 1 - x**(k+1)
Sk = 1 - x**(k+1)  / 1 - x ; x -> [0, 1], k>=0 ;  
Sk = 1 / 1 - x ; k -> OO
Sk' = 1 + 2*x + 3*x^2 + ... + k * x**(k-1) = 1 / (1-x)^2
xSk' = 1*x^1 + 2*x^2 + 3*x^3 + .... + k * x^k = x / (1-x)^2

Ek = 1 - 1/n+1 + 1 - 1/n+2 + ... + 1-1/n+k = k - Sk(1/n+k) 
 
ssr (p)
nor ssr (q, p)
nor nor ssr (q, q, p) 
...
nor nor nor ... ssr

令 X = 第k次抽到 SSR 之前， 抽到i次普通装备
P(Xi) = q^i * p
EP = S( i * P(Xi) ) = i * p * q ^ i = p * (i*q^i)
   = 0*q^0*p + 1*q^1*p + 2*q^2*p + q^3*p + ... k*q^k*p = p(1*q^1+2*q^2+...k*q^k)  = p * q / (1-q)^2 = pq / p**2 
   = q / p
EP = S( i * P(Xi) ) = q / p =  n / k

S(EP) =  1 * (n / k)  + 2 [k is left numbers of ssr, [1, m] ] 
EXP = sum (2+n/k for k in range(1, m+1)) = 2*m + H(m) = 2*m + (1+1/2+...1/m-1+1/m)

'''


'''
输入例子：
2 1
输出例子：
4.00

P = 1 / 3

2      1/3              2. 
1 2    (1-1/3) * 1/3    1+2
1 1 2  (1-1/3)^2 * 1/3  1*2+2
...
1 1 . 2  (1-1/3)^k * 1/3  1*k+2
Exp = 1/3 * 2 + (1-1/3) * 1/3 * (1+2) + (1-1/3)^2 * 1/3  * (1*2+2) ... + (1-1/3)^k * 1/3 * (1*k+2)
    = sum((1-1/3)^i * 1/3 * (1*i+2) for i in range(k) ) [k -> OO]
    = fn ( 2^i * (i+2) / 3^(i+1) ) (i>=0)

输入例子：
2 2
输出例子：
7.00

p = 2 / 4 = 1 / 2 = m / (m + n)

2 2    p ^ m          2 * m 
1 2 2  (1-p) * p ^ m       1 + 2*m
1 1 2 2 Cr(2+m-1, 2)*(1-p)^2 * p ^ m     1*2 + 2*m
...
1 1 . 2  Cr(k+m-1, k)*(1-p)^k * p ^ m   1*k + 2*m

Exp = p ^ m  * 2*m + (1-p) * p ^ m  * (1+2*m) + (1-p)^2 * p ^ m * (1*2+2*m) ... + (1-p)^k * p ^ m * (1*k+2*m)
    = sum( Cr(i+m-1, i) * (m/m+n)^i * (n/m+n) ^ m * (1*i+2*m) for i in range(k) ) [k -> OO]
    = sum( fn( Cr(i+m-1, i) *  m^i*n^m * (i+2*m) / (m+n)^(m+i) ) (i>=0) )



输入例子：
5 6
输出例子：
24.25


m * 2 + n 
m / (m + n) + (m - 1) / (m - 1 + n) 

[2, 2, 2, 2, ... 2] -> p0 = ( m / (m + n) * (m-1) / (m+n-1) * ... * 1/ n+1 ) ,  m * 2; 
[2, 2, 2, 2, ... 2, 1, 2] ->  Cr(m, 1) * p0 , m * 2 + 1
[2, 2, 2, 2, ... 2, 1, 1, 2] -> Cr(m+1, 2) * p0 , m * 2 + 2
... 
[2, 2, 2, 2, 1, ... 1, 1, 2] ->  Cr(m+n-1, n) * p0 , m * 2 + n
...
[2, 2, 2, 2, 1, ... 1, 1, 2] ->  Cr(m+n+oo-1, n) * p0 , m * 2 + n + oo -> Cr(n-1, n) * p0 , n

'''

''' 
p0 = 1
for i in range(1, m+1):
    p0 *= i / (n+i)

def Cr(N, K):
    if K == 0: return 1
    N_ = 1
    for num in range(2, N+1):
        N_ *= num
    K_ = 1
    for num in range(2, K+1):
        K_ *= num
    N_K_ = 1
    for num in range(2, N-K+1):
        N_K_ *= num

    return N_ / (K_ * N_K_) 


sum1 = sum2 = 0
for j in range(n*10+1):
    sum1 += Cr(m+j-1, j) * p0 * (m * 2 + j)
    sum2 += Cr(m+j-1, j) * p0
sum3 = 0
for i in range(10):
    sum3 += Cr(i+m-1, i) * ( (m**i) * (n**m) * (i+2*m) ) / ( (m+n)**(m+i) )

print(Cr(5, 0), 1/Cr(5, 1), 1/Cr(5, 2), 1/Cr(5, 3), Cr(5, 5), sum1, sum2)

exp = sum3

print(exp)

'''
