'''
题目大意（Number of Atoms）
给定一个化学式（字符串），例如 "H2O"、"Mg(OH)2"、"K4(ON(SO3)2)2"，要求解析出每种原子的数量，按字典序返回格式化后的字符串。

✅ 示例
python
复制
编辑
Input: "H2O"
Output: "H2O"

Input: "Mg(OH)2"
Output: "H2MgO2"

Input: "K4(ON(SO3)2)2"
Output: "K4N2O14S4"
'''
# O(N) O(N)
from collections import defaultdict

def count_of_atoms(S):
    if not S: return ""
    i = 0
    stack = [ defaultdict(int) ]
    N = len(S)

    while i < N:
        if S[i] == '(':
            i += 1
            stack.append(defaultdict(int))

        elif S[i] == ')':
            i += 1
            atom_freq_dict = stack.pop()

            num = 0
            while i < N and S[i].isdigit():
                num = 10 * num + int(S[i])
                i += 1
            num = max(num, 1)
            for atom in atom_freq_dict:
                stack[-1][atom] += num * atom_freq_dict[atom]

        elif S[i].isupper():
             atom = S[i]
             i += 1

             while i < N and S[i].islower():
                 atom += S[i]
                 i += 1

             num = 0
             while i < N and S[i].isdigit():
                 num = 10 * num + int(S[i])
                 i += 1
             num = max(num, 1)

             stack[-1][atom] += num
        else:
            return False

    res = ""

    for atom, freq in sorted(stack[-1].items()):
        res += atom + (str(freq) if freq > 1 else "")

    return res

print(count_of_atoms(""))
print(count_of_atoms("H2O"))
print(count_of_atoms("Mg(OH)2"))
print(count_of_atoms("K4(ON(SO3)2)2"))