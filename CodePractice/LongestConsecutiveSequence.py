def longest_increased_1_subarray(A):
    L = [A[0]]
    res = L[:]
    mx = cur = 1
    for i in range(1, len(A)):
        if A[i-1] + 1 == A[i]:
            cur += 1
            L.append(A[i])
        else:
            if mx < cur:
                mx = cur
                res = L[:]
            cur = 1
            L = [A[i]]

    if mx < cur:
        mx = cur
        res = L[:]
    return res, mx


def longest_increased_subarray(A):
    L = [A[0]]
    res = L[:]
    mx = cur = 1
    for i in range(1, len(A)):
        if A[i - 1]  < A[i]:
            cur += 1
            L.append(A[i])
        else:
            if mx < cur:
                mx = cur
                res = L[:]
            cur = 1
            L = [A[i]]
    if mx < cur:
        mx = cur
        res = L[:]
    return res, mx


def longest_increased_1_sequence(A):
    pass

def longest_increased_sequence(A):
    pass



print(longest_increased_1_subarray([1,2,3,6,7,8,9, 192,191,195]))
print(longest_increased_subarray([1,2,3,6,7,8,9, 192,191,195]))
print(longest_increased_1_sequence([1, 2, 3, 6, 7, 8, 9, 192, 191, 195]))
print(longest_increased_sequence([1, 2, 3, 6, 7, 8, 9, 192, 191, 195]))
