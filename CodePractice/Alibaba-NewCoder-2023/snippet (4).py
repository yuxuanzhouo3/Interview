import sys
from bisect import bisect_left, bisect_right

def solve():
    input = sys.stdin.readline

    n = int(input())

    people = []

    for idx in range(n):
        A, B = map(int, input().split())
        people.append((A - B, A, B, idx))

    # 按 D = A - B 排序
    people.sort()

    d = [x[0] for x in people]

    # prefix[k]：
    # people[0..k] 中 A 最大和第二大的 (A, 原始编号)
    prefix = [None] * n

    best1 = (-1, -1)
    best2 = (-1, -1)

    for i in range(n):
        A = people[i][1]
        idx = people[i][3]

        cur = (A, idx)

        if cur[0] > best1[0]:
            best2 = best1
            best1 = cur
        elif cur[0] > best2[0]:
            best2 = cur

        prefix[i] = (best1, best2)

    # suffix[k]：
    # people[k..n-1] 中 B 最大和第二大的 (B, 原始编号)
    suffix = [None] * n

    best1 = (-1, -1)
    best2 = (-1, -1)

    for i in range(n - 1, -1, -1):
        B = people[i][2]
        idx = people[i][3]

        cur = (B, idx)

        if cur[0] > best1[0]:
            best2 = best1
            best1 = cur
        elif cur[0] > best2[0]:
            best2 = cur

        suffix[i] = (best1, best2)

    ans = 0

    for _, A, B, idx in people:
        D = A - B

        # 情况1：
        # Dj <= -Di
        # 此时 Ai + Aj <= Bi + Bj
        # 答案由 A 决定
        pos = bisect_right(d, -D) - 1

        if pos >= 0:
            first, second = prefix[pos]

            if first[1] != idx:
                ans = max(ans, A + first[0])
            elif second[1] != -1:
                ans = max(ans, A + second[0])

        # 情况2：
        # Dj >= -Di
        # 此时 Ai + Aj >= Bi + Bj
        # 答案由 B 决定
        pos = bisect_left(d, -D)

        if pos < n:
            first, second = suffix[pos]

            if first[1] != idx:
                ans = max(ans, B + first[0])
            elif second[1] != -1:
                ans = max(ans, B + second[0])

    # 原题是平均值，需要 / 2
    # 根据题目要求决定输出格式
    if ans % 2 == 0:
        print(ans // 2 * 1.0)
    else:
        print(ans / 2 * 1.0)


if __name__ == "__main__":
    solve()