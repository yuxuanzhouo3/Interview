import collections

def check_binary_tree(edges):
    children = collections.defaultdict(set)
    indegree = {}
    v = set()

    for (p, c) in edges:
        if (p, c) in v:
            return "Error: Duplicate Edge"
        v.add((p,c))
        if p not in indegree:
            indegree[p] = 0
        if c not in indegree:
            indegree[c] = 0
        children[p].add(c)
        indegree[c] += 1
        if indegree[c] > 1:
            return "Error: Child has multiple parents"
        if len(children[p]) > 2:
            return "Error: Node has more than two children"

    q = [ node for node, in_degree in indegree.items() if in_degree == 0]
    if not q:
        return "Error: Cycle detected"
    if len(q) > 1:
        return "Error: Multiple roots found"

    # BFS to detect cycle
    seen = set([q[0]])
    while q:
        p = q.pop(0)
        for c in children[p]:
            if c in seen:
                return "Error: Cycle detected"
            seen.add(c)
            indegree[c] -= 1
            if indegree[c] == 0:
                q.append(c)

    return "Valid binary tree" if len(seen) == len(indegree.keys()) else "Error: Cycle detected"

# 示例调用
edges1 = [('A', 'B'), ('A', 'C'), ('B', 'D')]
print(check_binary_tree(edges1))  # Valid binary tree

edges2 = [('A', 'B'), ('A', 'C'), ('A', 'D')]  # 3 children
print(check_binary_tree(edges2))  # Error: Node has more than two children

edges3 = [('A', 'B'), ('B', 'A')]  # cycle
print(check_binary_tree(edges3))  # Error: Cycle detected

edges4 = [('A', 'B'), ('C', 'B')]  # multiple parents
print(check_binary_tree(edges4))  # Error: Child has multiple parents

edges5 = [('A', 'B'), ('C', 'D')]  # multiple roots
print(check_binary_tree(edges5))  # Error: Multiple roots found