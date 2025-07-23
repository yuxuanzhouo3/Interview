# 1. Stack
class Stack:
    def __init__(self): self.data = []

    def push(self, x): self.data.append(x)

    def pop(self): return self.data.pop()

    def peek(self): return self.data[-1] if self.data else None

    def empty(self): return not self.data


# 2. Queue
class Queue:
    def __init__(self): self.data = []

    def enqueue(self, x): self.data.append(x)

    def dequeue(self): return self.data.pop(0)

    def peek(self): return self.data[0] if self.data else None

    def empty(self): return not self.data


# 3. CircularQueue
class CircularQueue:
    def __init__(self, k):
        self.q, self.k, self.head, self.count = [0] * k, k, 0, 0

    def enQueue(self, value):
        if self.isFull(): return False
        tail = (self.head + self.count) % self.k
        self.q[tail] = value;
        self.count += 1;
        return True

    def deQueue(self):
        if self.isEmpty(): return False
        self.head = (self.head + 1) % self.k;
        self.count -= 1;
        return True

    def Front(self):
        return -1 if self.isEmpty() else self.q[self.head]

    def Rear(self):
        return -1 if self.isEmpty() else self.q[(self.head + self.count - 1) % self.k]

    def isEmpty(self):
        return self.count == 0

    def isFull(self):
        return self.count == self.k


# 4. Linked List Node
class ListNode:
    def __init__(self, val=0, nxt=None): self.val, self.next = val, nxt


# 5. LinkedList
class LinkedList:
    def __init__(self):
        self.head = None

    def insert_front(self, val):
        self.head = ListNode(val, self.head)

    def search(self, val):
        cur = self.head
        while cur:
            if cur.val == val: return True
            cur = cur.next
        return False


# 6. Doubly Linked List Node
class DListNode:
    def __init__(self, val=0): self.val, self.prev, self.next = val, None, None


# 7. Binary Tree Node
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right


# 8. Binary Search Tree
class BST:
    def __init__(self):
        self.root = None

    def insert(self, val):
        def _insert(node, val):
            if not node: return TreeNode(val)
            if val < node.val:
                node.left = _insert(node.left, val)
            else:
                node.right = _insert(node.right, val)
            return node

        self.root = _insert(self.root, val)

    def inorder(self):
        res = []

        def dfs(node):
            if not node: return
            dfs(node.left);
            res.append(node.val);
            dfs(node.right)

        dfs(self.root);
        return res


# 9. AVL Node
class AVLNode:
    def __init__(self, val=0):
        self.val, self.height, self.left, self.right = val, 1, None, None


# 10. Graph
class Graph:
    def __init__(self): self.adj = {}

    def add_edge(self, u, v):
        self.adj.setdefault(u, []).append(v)
        self.adj.setdefault(v, []).append(u)


# 11. Trie Node
class TrieNode:
    def __init__(self):
        self.children, self.isWord = {}, False


# 12. Trie
class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for ch in word: node = node.children.setdefault(ch, TrieNode())
        node.isWord = True

    def search(self, word):
        node = self.root
        for ch in word:
            if ch not in node.children: return False
            node = node.children[ch]
        return node.isWord


# 13. DSU (Union-Find)
class DSU:
    def __init__(self, n):
        self.parent = list(range(n)); self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x: self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        xr, yr = self.find(x), self.find(y)
        if xr == yr: return
        if self.rank[xr] < self.rank[yr]:
            self.parent[xr] = yr
        elif self.rank[xr] > self.rank[yr]:
            self.parent[yr] = xr
        else:
            self.parent[yr] = xr; self.rank[xr] += 1


# 14. Heap (MinHeap wrapper)
import heapq


class MinHeap:
    def __init__(self): self.h = []

    def push(self, x): heapq.heappush(self.h, x)

    def pop(self): return heapq.heappop(self.h)

    def peek(self): return self.h[0]


# 15. MaxHeap (via negative)
class MaxHeap:
    def __init__(self): self.h = []

    def push(self, x): heapq.heappush(self.h, -x)

    def pop(self): return -heapq.heappop(self.h)

    def peek(self): return -self.h[0]


# 16. PriorityQueue
class PriorityQueue:
    def __init__(self): self.h = []

    def push(self, x, p): heapq.heappush(self.h, (p, x))

    def pop(self): return heapq.heappop(self.h)[1]


# 17. LRU Cache (OrderedDict)
from collections import OrderedDict


class LRUCacheOrderedDict:
    def __init__(self, capacity: int):
        self.cache, self.cap = OrderedDict(), capacity

    def get(self, key):
        if key not in self.cache: return -1
        self.cache.move_to_end(key);
        return self.cache[key]

    def put(self, key, val):
        if key in self.cache: self.cache.move_to_end(key)
        self.cache[key] = val
        if len(self.cache) > self.cap: self.cache.popitem(last=False)


# 18. LFU Cache Node
class LFUNode:
    def __init__(self, key, val, freq=1): self.key, self.val, self.freq = key, val, freq


# 19. Interval class
class Interval:
    def __init__(self, start, end): self.start, self.end = start, end

    def __lt__(self, other): return self.start < other.start


# 20. Point class
class Point:
    def __init__(self, x=0, y=0): self.x, self.y = x, y


# 21. Rectangle class
class Rectangle:
    def __init__(self, width, height): self.width, self.height = width, height

    def area(self): return self.width * self.height

    def perimeter(self): return 2 * (self.width + self.height)


# 22. Circle class
import math


class Circle:
    def __init__(self, radius): self.radius = radius

    def area(self): return math.pi * self.radius ** 2

    def perimeter(self): return 2 * math.pi * self.radius


# 23. Matrix
class Matrix:
    def __init__(self, data): self.data = data

    def transpose(self):
        return [[self.data[j][i] for j in range(len(self.data))]
                for i in range(len(self.data[0]))]

    def add(self, other):
        return [[self.data[i][j] + other.data[i][j]
                 for j in range(len(self.data[0]))]
                for i in range(len(self.data))]


# 24. SparseMatrix (dict-based)
class SparseMatrix:
    def __init__(self):
        self.data = {}

    def set(self, r, c, val):
        if val != 0:
            self.data[(r, c)] = val
        elif (r, c) in self.data:
            del self.data[(r, c)]

    def get(self, r, c):
        return self.data.get((r, c), 0)


# 25. Vector2D
class Vector2D:
    def __init__(self, x=0, y=0): self.x, self.y = x, y

    def add(self, other): return Vector2D(self.x + other.x, self.y + other.y)


# 26. Vector3D
class Vector3D:
    def __init__(self, x=0, y=0, z=0): self.x, self.y, self.z = x, y, z

    def cross(self, other):
        return Vector3D(self.y * other.z - self.z * other.y,
                        self.z * other.x - self.x * other.z,
                        self.x * other.y - self.y * other.x)


# 27. ComplexNumber
class ComplexNumber:
    def __init__(self, real, imag): self.real, self.imag = real, imag

    def add(self, other): return ComplexNumber(self.real + other.real, self.imag + other.imag)

    def mul(self, other):
        return ComplexNumber(self.real * other.real - self.imag * other.imag,
                             self.real * other.imag + self.imag * other.real)


# 28. Polynomial
class Polynomial:
    def __init__(self, coeffs): self.coeffs = coeffs  # list of coeff

    def eval(self, x): return sum(c * (x ** i) for i, c in enumerate(self.coeffs))


# 29. Fraction class
from math import gcd


class Fraction:
    def __init__(self, num, den):
        g = gcd(num, den);
        self.num, self.den = num // g, den // g

    def __add__(self, other):
        return Fraction(self.num * other.den + self.den * other.num, self.den * other.den)


# 30. BigInteger (simple string add)
class BigInteger:
    def __init__(self, val): self.val = val

    def add(self, other):
        return BigInteger(str(int(self.val) + int(other.val)))


# 31. Bitset
class Bitset:
    def __init__(self, size):
        self.size, self.bits = size, 0

    def set(self, idx): self.bits |= 1 << idx

    def unset(self, idx): self.bits &= ~(1 << idx)

    def get(self, idx): return (self.bits >> idx) & 1


# 32. BloomFilter (simple hash set wrapper)
class BloomFilter:
    def __init__(self): self.s = set()

    def add(self, key): self.s.add(hash(key))

    def contains(self, key): return hash(key) in self.s


# 33. HashMap
class HashMap:
    def __init__(self): self.d = {}

    def put(self, k, v): self.d[k] = v

    def get(self, k): return self.d.get(k, None)

    def remove(self, k): self.d.pop(k, None)


# 34. HashSet
class HashSet:
    def __init__(self): self.s = set()

    def add(self, x): self.s.add(x)

    def contains(self, x): return x in self.s

    def remove(self, x): self.s.discard(x)


# 35. SegmentTree (sum)
class SegmentTree:
    def __init__(self, nums):
        self.n = len(nums)
        self.tree = [0] * (2 * self.n)
        for i in range(self.n): self.tree[self.n + i] = nums[i]
        for i in range(self.n - 1, 0, -1): self.tree[i] = self.tree[i << 1] + self.tree[i << 1 | 1]

    def update(self, idx, val):
        pos = idx + self.n;
        self.tree[pos] = val
        while pos > 1: pos >>= 1; self.tree[pos] = self.tree[pos << 1] + self.tree[pos << 1 | 1]

    def query(self, l, r):
        res = 0;
        l += self.n;
        r += self.n
        while l < r:
            if l & 1: res += self.tree[l]; l += 1
            if r & 1: r -= 1; res += self.tree[r]
            l >>= 1;
            r >>= 1
        return res


# 36. Fenwick Tree (BIT)
class FenwickTree:
    def __init__(self, n):
        self.n, self.tree = n, [0] * (n + 1)

    def update(self, i, delta):
        while i <= self.n: self.tree[i] += delta; i += i & -i

    def query(self, i):
        res = 0
        while i > 0: res += self.tree[i]; i -= i & -i
        return res


# 37. KDTree Node
class KDNode:
    def __init__(self, point, left=None, right=None):
        self.point, self.left, self.right = point, left, right


# 38. KDTree (placeholder)
class KDTree:
    def __init__(self, points): self.root = None  # actual build skipped

    def nearest(self, target): return None  # placeholder


# 39. DiscreteEventSimulator (basic)
import heapq


class DiscreteEventSimulator:
    def __init__(self): self.events = []

    def schedule(self, time, action): heapq.heappush(self.events, (time, action))

    def run(self):
        while self.events:
            t, act = heapq.heappop(self.events)
            act()


# 40. Logger (OO from LC 359)
class Logger:
    def __init__(self): self.lastPrint = {}

    def shouldPrintMessage(self, timestamp, message):
        if message not in self.lastPrint or timestamp - self.lastPrint[message] >= 10:
            self.lastPrint[message] = timestamp;
            return True
        return False


# 41. SkipList Node
class SkipListNode:
    def __init__(self, val, level):
        self.val = val
        self.forward = [None] * (level + 1)


# 42. SkipList
import random


class SkipList:
    MAX_LEVEL = 4
    P = 0.5

    def __init__(self):
        self.header = SkipListNode(-1, self.MAX_LEVEL)
        self.level = 0

    def random_level(self):
        lvl = 0
        while random.random() < self.P and lvl < self.MAX_LEVEL:
            lvl += 1
        return lvl

    def insert(self, val):
        update = [None] * (self.MAX_LEVEL + 1)
        x = self.header
        for i in reversed(range(self.level + 1)):
            while x.forward[i] and x.forward[i].val < val:
                x = x.forward[i]
            update[i] = x
        x = x.forward[0]
        if x is None or x.val != val:
            lvl = self.random_level()
            if lvl > self.level:
                for i in range(self.level + 1, lvl + 1):
                    update[i] = self.header
                self.level = lvl
            newNode = SkipListNode(val, lvl)
            for i in range(lvl + 1):
                newNode.forward[i] = update[i].forward[i]
                update[i].forward[i] = newNode

    def search(self, val):
        x = self.header
        for i in reversed(range(self.level + 1)):
            while x.forward[i] and x.forward[i].val < val:
                x = x.forward[i]
        x = x.forward[0]
        return x and x.val == val


# 43. EventBus
class EventBus:
    def __init__(self): self.subscribers = {}

    def subscribe(self, event, callback):
        self.subscribers.setdefault(event, []).append(callback)

    def publish(self, event, data):
        for callback in self.subscribers.get(event, []): callback(data)


# 44. RateLimiter (Token Bucket)
import time


class RateLimiter:
    def __init__(self, rate, capacity):
        self.rate, self.capacity = rate, capacity
        self.tokens = capacity
        self.last = time.time()

    def allow(self):
        now = time.time()
        delta = now - self.last
        self.last = now
        self.tokens = min(self.capacity, self.tokens + delta * self.rate)
        if self.tokens >= 1:
            self.tokens -= 1;
            return True
        return False


# 45. LRU Cache (Double Linked List)
class Node:
    def __init__(self, k, v):
        self.k, self.v, self.prev, self.next = k, v, None, None


class LRUCacheDoubleLinkedList:
    def __init__(self, capacity):
        self.cap = capacity;
        self.map = {}
        self.head, self.tail = Node(0, 0), Node(0, 0)
        self.head.next, self.tail.prev = self.tail, self.head

    def _remove(self, node):
        p, n = node.prev, node.next
        p.next, n.prev = n, p

    def _insert(self, node):
        n = self.head.next
        self.head.next = node;
        node.prev = self.head
        node.next = n;
        n.prev = node

    def get(self, k):
        if k in self.map:
            self._remove(self.map[k]);
            self._insert(self.map[k])
            return self.map[k].v
        return -1

    def put(self, k, v):
        if k in self.map: self._remove(self.map[k])
        self.map[k] = Node(k, v);
        self._insert(self.map[k])
        if len(self.map) > self.cap:
            n = self.tail.prev;
            self._remove(n);
            del self.map[n.k]


# 46. LFU Cache (简化)
from collections import defaultdict, OrderedDict


class LFUCache:
    def __init__(self, capacity):
        self.cap = capacity;
        self.minFreq = 0
        self.keyToValFreq = {};
        self.freqToKeys = defaultdict(OrderedDict)

    def get(self, key):
        if key not in self.keyToValFreq: return -1
        val, freq = self.keyToValFreq[key]
        del self.freqToKeys[freq][key]
        if not self.freqToKeys[freq]: del self.freqToKeys[freq]
        self.freqToKeys[freq + 1][key] = None
        self.keyToValFreq[key] = (val, freq + 1)
        self.minFreq = min(self.minFreq, freq + 1)
        return val

    def put(self, key, value):
        if self.cap == 0: return
        if key in self.keyToValFreq:
            _, freq = self.keyToValFreq[key]
            self.keyToValFreq[key] = (value, freq)
            self.get(key);
            return
        if len(self.keyToValFreq) >= self.cap:
            k, _ = self.freqToKeys[self.minFreq].popitem(last=False)
            del self.keyToValFreq[k]
        self.keyToValFreq[key] = (value, 1)
        self.freqToKeys[1][key] = None
        self.minFreq = 1


# 47. MedianFinder (LC 295)
import heapq


class MedianFinder:
    def __init__(self):
        self.small, self.large = [], []

    def addNum(self, num):
        heapq.heappush(self.small, -num)
        if self.small and self.large and -self.small[0] > self.large[0]:
            heapq.heappush(self.large, -heapq.heappop(self.small))
        if len(self.small) > len(self.large) + 1:
            heapq.heappush(self.large, -heapq.heappop(self.small))
        if len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappop(self.large))

    def findMedian(self):
        if len(self.small) > len(self.large): return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2


# 48. MovingAverage
class MovingAverage:
    def __init__(self, size):
        self.size, self.q, self.sum = size, [], 0

    def next(self, val):
        self.q.append(val);
        self.sum += val
        if len(self.q) > self.size: self.sum -= self.q.pop(0)
        return self.sum / len(self.q)


# 49. HitCounter
class HitCounter:
    def __init__(self): self.hits = []

    def hit(self, timestamp): self.hits.append(timestamp)

    def getHits(self, timestamp):
        while self.hits and timestamp - self.hits[0] >= 300: self.hits.pop(0)
        return len(self.hits)


# 50. SnakeGame (LC 353)
class SnakeGame:
    def __init__(self, width, height, food):
        self.width, self.height, self.food = width, height, food
        self.score = 0;
        self.body = [(0, 0)];
        self.foodIndex = 0

    def move(self, direction):
        head = self.body[0];
        dx, dy = 0, 0
        if direction == "U":
            dx, dy = -1, 0
        elif direction == "D":
            dx, dy = 1, 0
        elif direction == "L":
            dx, dy = 0, -1
        else:
            dx, dy = 0, 1
        newHead = (head[0] + dx, head[1] + dy)
        if (newHead in self.body[:-1] or
                not (0 <= newHead[0] < self.height) or
                not (0 <= newHead[1] < self.width)): return -1
        self.body.insert(0, newHead)
        if (self.foodIndex < len(self.food) and newHead == tuple(self.food[self.foodIndex])):
            self.score += 1;
            self.foodIndex += 1
        else:
            self.body.pop()
        return self.score


# 51. TicTacToe (LC 348)
class TicTacToe:
    def __init__(self, n):
        self.rows = [0] * n;
        self.cols = [0] * n;
        self.diag = self.anti = 0;
        self.n = n

    def move(self, row, col, player):
        add = 1 if player == 1 else -1
        self.rows[row] += add;
        self.cols[col] += add
        if row == col: self.diag += add
        if row + col == self.n - 1: self.anti += add
        if (abs(self.rows[row]) == self.n or abs(self.cols[col]) == self.n
                or abs(self.diag) == self.n or abs(self.anti) == self.n): return player
        return 0


# 52. RandomizedSet (LC 380)
class RandomizedSet:
    def __init__(self):
        self.d = {}; self.arr = []

    def insert(self, val):
        if val in self.d: return False
        self.d[val] = len(self.arr);
        self.arr.append(val);
        return True

    def remove(self, val):
        if val not in self.d: return False
        idx = self.d[val];
        last = self.arr[-1]
        self.arr[idx] = last;
        self.d[last] = idx
        self.arr.pop();
        del self.d[val];
        return True

    def getRandom(self):
        return random.choice(self.arr)


# 53. RandomizedCollection (LC 381)
class RandomizedCollection:
    def __init__(self): self.valToIdxs = defaultdict(set); self.arr = []

    def insert(self, val):
        self.arr.append(val);
        self.valToIdxs[val].add(len(self.arr) - 1)
        return len(self.valToIdxs[val]) == 1

    def remove(self, val):
        if val not in self.valToIdxs or not self.valToIdxs[val]: return False
        idx = self.valToIdxs[val].pop();
        last = self.arr[-1]
        self.arr[idx] = last;
        self.valToIdxs[last].add(idx)
        self.valToIdxs[last].discard(len(self.arr) - 1);
        self.arr.pop()
        return True

    def getRandom(self): return random.choice(self.arr)


# 54. PeekingIterator (LC 284)
class PeekingIterator:
    def __init__(self, iterator):
        self.iterator = iterator;
        self.peeked = None

    def peek(self):
        if self.peeked is None: self.peeked = next(self.iterator)
        return self.peeked

    def next(self):
        if self.peeked is not None:
            val = self.peeked;
            self.peeked = None;
            return val
        return next(self.iterator)

    def hasNext(self):
        return self.peeked is not None or self.iterator.hasNext()


# 55. PhoneDirectory (LC 379)
class PhoneDirectory:
    def __init__(self, maxNumbers):
        self.available = set(range(maxNumbers))

    def get(self):
        return self.available.pop() if self.available else -1

    def check(self, number):
        return number in self.available

    def release(self, number):
        self.available.add(number)


# 56. AutocompleteSystem (LC 642 simplified)
class AutocompleteSystem:
    def __init__(self, sentences, times):
        self.data = {s: t for s, t in zip(sentences, times)};
        self.cur = ""

    def input(self, c):
        if c == "#": self.data[self.cur] = self.data.get(self.cur, 0) + 1; self.cur = ""; return []
        self.cur += c
        return sorted([s for s in self.data if s.startswith(self.cur)],
                      key=lambda x: (-self.data[x], x))[:3]


# 57. FileSystem (LC 588 simplified)
class FileSystem:
    def __init__(self):
        self.fs = {"/": {}}

    def ls(self, path):
        parts = path.strip("/").split("/")
        node = self.fs["/"]
        for p in parts:
            if p: node = node[p]
        return sorted(node.keys()) if isinstance(node, dict) else [path.split("/")[-1]]

    def mkdir(self, path):
        parts = path.strip("/").split("/")
        node = self.fs["/"]
        for p in parts:
            if p not in node: node[p] = {}
            node = node[p]

    def addContentToFile(self, filePath, content):
        parts = filePath.strip("/").split("/")
        node = self.fs["/"]
        for p in parts[:-1]: node = node[p]
        node[parts[-1]] = node.get(parts[-1], "") + content

    def readContentFromFile(self, filePath):
        parts = filePath.strip("/").split("/")
        node = self.fs["/"]
        for p in parts[:-1]: node = node[p]
        return node[parts[-1]]


# 58. Leaderboard (LC 1244)
class Leaderboard:
    def __init__(self): self.scores = defaultdict(int)

    def addScore(self, playerId, score): self.scores[playerId] += score

    def top(self, K): return sum(sorted(self.scores.values(), reverse=True)[:K])

    def reset(self, playerId): self.scores[playerId] = 0


# 59. TimeMap (LC 981)
class TimeMap:
    def __init__(self):
        self.d = defaultdict(list)

    def set(self, key, value, timestamp):
        self.d[key].append((timestamp, value))

    def get(self, key, timestamp):
        arr = self.d.get(key, []);
        l, r = 0, len(arr) - 1;
        ans = ""
        while l <= r:
            m = (l + r) // 2
            if arr[m][0] <= timestamp:
                ans = arr[m][1]; l = m + 1
            else:
                r = m - 1
        return ans


# 60. ExamRoom (LC 855)
class ExamRoom:
    def __init__(self, N):
        self.N = N; self.seats = []

    def seat(self):
        if not self.seats: self.seats.append(0); return 0
        self.seats.sort();
        dist = self.seats[0];
        pos = 0
        for i in range(len(self.seats) - 1):
            d = (self.seats[i + 1] - self.seats[i]) // 2
            if d > dist: dist, pos = d, (self.seats[i] + self.seats[i + 1]) // 2
        if self.N - 1 - self.seats[-1] > dist: pos = self.N - 1
        self.seats.append(pos);
        return pos

    def leave(self, p):
        self.seats.remove(p)

import collections
import random

# 61. Copy List with Random Pointer Node
class RandomListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
        self.random = None

# 62. Copy List with Random Pointer (Deep copy)
def copyRandomList(head):
    if not head: return None
    # 1. Clone nodes and insert next to original
    cur = head
    while cur:
        nxt = cur.next
        copy = RandomListNode(cur.val)
        cur.next = copy
        copy.next = nxt
        cur = nxt
    # 2. Copy random pointers
    cur = head
    while cur:
        if cur.random:
            cur.next.random = cur.random.next
        cur = cur.next.next
    # 3. Separate lists
    cur = head
    copyHead = head.next
    while cur:
        copy = cur.next
        cur.next = copy.next
        copy.next = copy.next.next if copy.next else None
        cur = cur.next
    return copyHead

# 63. Flatten Nested List Iterator (LC 341)
class NestedInteger:
    # Simulated interface
    def isInteger(self): pass
    def getInteger(self): pass
    def getList(self): pass

class NestedIterator:
    def __init__(self, nestedList):
        self.stack = []
        self._pushList(nestedList)

    def _pushList(self, nestedList):
        for i in reversed(nestedList):
            self.stack.append(i)

    def next(self):
        if self.hasNext():
            return self.stack.pop().getInteger()

    def hasNext(self):
        while self.stack:
            top = self.stack[-1]
            if top.isInteger():
                return True
            self.stack.pop()
            self._pushList(top.getList())
        return False

# 64. Flatten a multilevel doubly linked list (LC 430)
class DNode:
    def __init__(self, val, prev=None, next=None, child=None):
        self.val, self.prev, self.next, self.child = val, prev, next, child

def flatten(head):
    if not head: return head
    pseudoHead = DNode(0, None, head, None)
    prev = pseudoHead

    stack = [head]
    while stack:
        curr = stack.pop()
        prev.next = curr
        curr.prev = prev

        if curr.next:
            stack.append(curr.next)
        if curr.child:
            stack.append(curr.child)
            curr.child = None

        prev = curr
    pseudoHead.next.prev = None
    return pseudoHead.next

# 65. Clone Graph Node
class GraphNode:
    def __init__(self, val):
        self.val = val
        self.neighbors = []

# 66. Clone Graph (DFS)
def cloneGraph(node):
    if not node: return None
    visited = {}

    def dfs(n):
        if n in visited:
            return visited[n]
        clone = GraphNode(n.val)
        visited[n] = clone
        for nei in n.neighbors:
            clone.neighbors.append(dfs(nei))
        return clone

    return dfs(node)

# 67. Clone Graph (BFS)
def cloneGraphBFS(node):
    if not node: return None
    visited = {node: GraphNode(node.val)}
    queue = collections.deque([node])
    while queue:
        n = queue.popleft()
        for nei in n.neighbors:
            if nei not in visited:
                visited[nei] = GraphNode(nei.val)
                queue.append(nei)
            visited[n].neighbors.append(visited[nei])
    return visited[node]

# 68. Trie with delete
class TrieNodeDel:
    def __init__(self):
        self.children = {}
        self.isWord = False

class TrieDel:
    def __init__(self):
        self.root = TrieNodeDel()

    def insert(self, word):
        node = self.root
        for ch in word:
            node = node.children.setdefault(ch, TrieNodeDel())
        node.isWord = True

    def search(self, word):
        node = self.root
        for ch in word:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return node.isWord

    def delete(self, word):
        def _delete(node, word, depth):
            if depth == len(word):
                if not node.isWord:
                    return False
                node.isWord = False
                return len(node.children) == 0
            ch = word[depth]
            if ch not in node.children:
                return False
            should_delete = _delete(node.children[ch], word, depth+1)
            if should_delete:
                del node.children[ch]
                return not node.isWord and len(node.children) == 0
            return False
        _delete(self.root, word, 0)

# 69. Flatten binary tree to linked list (LC 114)
def flattenBT(root):
    if not root: return
    flattenBT(root.left)
    flattenBT(root.right)

    left, right = root.left, root.right
    root.left = None
    root.right = left

    cur = root
    while cur.right:
        cur = cur.right
    cur.right = right

# 70. Copy Binary Tree (Deep copy)
def copyBT(root):
    if not root: return None
    newRoot = TreeNode(root.val)
    newRoot.left = copyBT(root.left)
    newRoot.right = copyBT(root.right)
    return newRoot

# 71. Flatten Nested List (Recursive)
def flattenList(nestedList):
    res = []
    for el in nestedList:
        if isinstance(el, list):
            res.extend(flattenList(el))
        else:
            res.append(el)
    return res

# 72. Flatten Nested List (Iterative)
def flattenListIter(nestedList):
    stack = nestedList[::-1]
    res = []
    while stack:
        el = stack.pop()
        if isinstance(el, list):
            stack.extend(el[::-1])
        else:
            res.append(el)
    return res

# 73. Binary Indexed Tree (Fenwick Tree) Range Update, Point Query
class FenwickTreeRangeUpdate:
    def __init__(self, n):
        self.n = n
        self.tree = [0]*(n+1)
    def update(self, idx, delta):
        while idx <= self.n:
            self.tree[idx] += delta
            idx += idx & (-idx)
    def range_update(self, l, r, delta):
        self.update(l, delta)
        self.update(r+1, -delta)
    def query(self, idx):
        res = 0
        while idx > 0:
            res += self.tree[idx]
            idx -= idx & (-idx)
        return res

# 74. Binary Indexed Tree (Fenwick Tree) Point Update, Range Query
class FenwickTreePointUpdateRangeQuery:
    def __init__(self, n):
        self.n = n
        self.tree = [0]*(n+1)
    def update(self, idx, delta):
        while idx <= self.n:
            self.tree[idx] += delta
            idx += idx & (-idx)
    def prefix_sum(self, idx):
        res = 0
        while idx > 0:
            res += self.tree[idx]
            idx -= idx & (-idx)
        return res
    def range_sum(self, l, r):
        return self.prefix_sum(r) - self.prefix_sum(l-1)

# 75. Weighted Union-Find (DSU with size tracking)
class WeightedUnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1]*n
    def find(self, x):
        while x != self.parent[x]:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.size[rx] < self.size[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        self.size[rx] += self.size[ry]
        return True

# 76. Tarjan's SCC Algorithm (Strongly Connected Components)
class TarjanSCC:
    def __init__(self, graph):
        self.graph = graph
        self.index = 0
        self.stack = []
        self.onStack = set()
        self.indices = {}
        self.lowlink = {}
        self.sccs = []

    def run(self):
        for node in self.graph:
            if node not in self.indices:
                self._strongconnect(node)
        return self.sccs

    def _strongconnect(self, v):
        self.indices[v] = self.index
        self.lowlink[v] = self.index
        self.index += 1
        self.stack.append(v)
        self.onStack.add(v)

        for w in self.graph[v]:
            if w not in self.indices:
                self._strongconnect(w)
                self.lowlink[v] = min(self.lowlink[v], self.lowlink[w])
            elif w in self.onStack:
                self.lowlink[v] = min(self.lowlink[v], self.indices[w])

        if self.lowlink[v] == self.indices[v]:
            scc = []
            while True:
                w = self.stack.pop()
                self.onStack.remove(w)
                scc.append(w)
                if w == v:
                    break
            self.sccs.append(scc)

# 77. Trie with Prefix Count
class TrieNodeCount:
    def __init__(self):
        self.children = {}
        self.count = 0
        self.isWord = False

class TrieCount:
    def __init__(self):
        self.root = TrieNodeCount()

    def insert(self, word):
        node = self.root
        for ch in word:
            node.children.setdefault(ch, TrieNodeCount())
            node = node.children[ch]
            node.count += 1
        node.isWord = True

    def countPrefix(self, prefix):
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return 0
            node = node.children[ch]
        return node.count

# 78. Topological Sort (Kahn's Algorithm)
def topologicalSort(graph):
    in_degree = {u: 0 for u in graph}
    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1
    queue = collections.deque([u for u in graph if in_degree[u] == 0])
    res = []
    while queue:
        u = queue.popleft()
        res.append(u)
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    if len(res) != len(graph):
        return []  # Cycle detected
    return res

# 79. KMP String Matching Algorithm
def kmp_search(text, pattern):
    def build_lps(p):
        lps = [0]*len(p)
        j = 0
        for i in range(1,len(p)):
            while j > 0 and p[i] != p[j]:
                j = lps[j-1]
            if p[i] == p[j]:
                j += 1
                lps[i] = j
        return lps
    lps = build_lps(pattern)
    j = 0
    for i in range(len(text)):
        while j > 0 and text[i] != pattern[j]:
            j = lps[j-1]
        if text[i] == pattern[j]:
            j += 1
        if j == len(pattern):
            return i - j + 1
    return -1

# 80. Rolling Hash (Rabin-Karp for substring search)
class RollingHash:
    def __init__(self, base=26, mod=10**9+7):
        self.base = base
        self.mod = mod
    def hash(self, s):
        h = 0
        for ch in s:
            h = (h * self.base + ord(ch)) % self.mod
        return h

