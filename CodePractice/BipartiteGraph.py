'''
给定一个 图，每个节点是一个人。 若 A 认识 B，说明 A 和 B 之间存在边。
我们需要将所有人分成 两组，使得 组内的人之间没有边（不认识）。
所以，我们要判断：是否可以将图的所有节点分成两个集合，使得所有边连接的两点都在不同集合中。
'''
from collections import deque

def can_be_two_groups(graph):
    N = len(graph)
    color = [0] * N # 0: not colored; 1: group A; -1: group B
    for i in range(N):
        if color[i]: continue
        q = [i]
        color[i] = 1
        while q:
            node = q.pop(0)
            for nei in graph[node]:
                if color[nei]:
                    if color[node] == color[nei]: return False
                else:
                    color[nei] = -color[node]
                    q.append(nei)
    print(color)
    return True


graph = [
    [1],     # 0 认识 1
    [0, 2],  # 1 认识 0 和 2
    [1]      # 2 认识 1
]

print(can_be_two_groups(graph))  # True：可以分为两组


graph = [
    [1, 2],  # 0 认识 1 和 2
    [0, 2],  # 1 认识 0 和 2
    [0, 1]   # 2 认识 0 和 1
]

print(can_be_two_groups(graph))  # False：三角结构，无法分为两组
