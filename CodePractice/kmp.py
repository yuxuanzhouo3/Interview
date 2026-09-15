def kmp(s):
    n = len(s)
    pi = [0] * n

    for i in range(1, n):

        # 上一个位置 i-1 已经匹配了 j 长度
        j = pi[i-1]

        # 当前字符不匹配, 尝试更短的 prefix 或 直到第一个字符
        while j > 0 and s[j] != s[i]:
            j = pi[j-1]

        # 当前字符匹配 【0...j-1】 match [i-j...i-1]; so s[j] == s[i] ??; if so, pi[i] = j+1
        if s[j] == s[i]:
            j += 1

        pi[i] = j

    return pi[-1], pi

res = ["abcdefgabc", "abcdefgabcd", "abcdefgabcde", "abcdefgabcdef", "abcdefgabcdefga"]
for r in res:
    print(kmp(r))

'''
KMP
│
├── Prefix Function / pi[]
│
│   pi[i] 
│   = s[0:i+1]
│     最长 proper prefix == suffix
│
├── match
│      s[i] == s[j]
│          ↓
│        j += 1
│
└── mismatch
       s[i] != s[j]
           ↓
       j = pi[j-1] # border's border, sub-border
       # [0, ..., j-1][j], ..., [i-j,... i-1] [i] 
       # [0,...,jj-1(B1)][jj] ..|B2, j-1][j], ..., [i-j, .B3| .[i-jj(B4),..., i-1] [i] 
       # B1 == B2, B3 == B4; B12 == B34;  ==> B1 == B3 == B4


pi = [0, 1, 0, 0, 0, 0, 1, 2, 3]

index:  0 1 2 3 4 5 6 7 8
s:      a a b f c g a a b

"a"          pi = 0

"aa"         prefix "a"
             suffix "a"
             pi = 1

"aab"        没有
             pi = 0

"aabf"       没有
             pi = 0

"aabfc"      没有
             pi = 0

"aabfcg"     没有
             pi = 0

"aabfcga"    prefix "a" == suffix "a"
             pi = 1

"aabfcgaa"   prefix "aa" == suffix "aa"
             pi = 2

"aabfcgaab"  prefix "aab" == suffix "aab"
             pi = 3

'''