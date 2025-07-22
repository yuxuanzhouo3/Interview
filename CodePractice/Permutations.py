nums = [1, 3, 4]
nums2 = [3, 4, 7, 1]

# O(N * N!) O(N)
def Permutations_backtracking(A):
    if not A: return []
    N = len(A)
    res = []
    seen = [0] * (N+1)

    def dfs_backtracking(cur):
        if len(cur) == N:
            res.append(cur[:])

        for j in range(N):
            if seen[j]:
                continue

            seen[j] = 1
            cur.append(A[j])
            dfs_backtracking(cur)
            cur.pop(-1)
            seen[j] = 0


    dfs_backtracking([])
    return res

# O(N * N!) O(N)
def Permutations_backtracking2(A):
    if not A: return []
    N = len(A)
    res = []
    # seen = [0] * (N+1)

    def dfs_backtracking2(cur, seen):
        if len(cur) == N:
            res.append(cur[:])

        for j in range(N):
            if seen[j]:
                continue

            seen[j] = 1
            cur.append(A[j])
            dfs_backtracking2(cur, seen)
            cur.pop(-1)
            seen[j] = 0


    dfs_backtracking2([], [0] * (N+1))
    return res


# O(N * N!) O(k)
def Permutations_backtracking3(A, k):
    if not A: return []
    N = len(A)
    res = []
    # seen = [0] * (N+1)

    def dfs_backtracking3(cur, seen, k, index):
        if len(cur) == k:
            res.append(cur[:])

        for j in range(index, N):
            if seen[j]:
                continue

            seen[j] = 1
            cur.append(A[j])
            dfs_backtracking3(cur, seen, k, j+1)
            cur.pop(-1)
            seen[j] = 0


    dfs_backtracking3([], [0] * (N+1), k, 0)
    return res


# O(N * N!)  O(N * N!)
def Permutations_iteration(A):
    # if not A or len(A) == 1: return A
    # N = len(A)
    res = [[]]

    for a in A:
        new_res = []
        for cur in res:
            for i in range(len(cur)+1):
                new_res.append(cur[:i] + [a] + cur[i:])
        res = new_res
    return res



from itertools import permutations
def permutations_unique(nums):
    return [list(p) for p in set(permutations(nums))]

from itertools import combinations
def combinations_unique(nums, k):
    return [list(p) for p in set(combinations(nums, k))]

print(Permutations_backtracking(nums))
print(Permutations_backtracking2(nums))
print(Permutations_backtracking3(nums, 2))
print(Permutations_iteration(nums))
print(permutations_unique(nums))
print(combinations_unique(nums, 2))
print(combinations(nums, 2))

print(Permutations_backtracking(nums2))
print(Permutations_backtracking2(nums2))
print(Permutations_backtracking3(nums2, 2))
print(Permutations_iteration(nums2))
print(permutations_unique(nums2))
print(combinations_unique(nums2, 2))
print(list(combinations(nums2, 2)))

