'''
528. Random Pick with Weight
You are given a 0-indexed array of positive integers w where w[i] describes the weight of the ith index.
You need to implement the function pickIndex(), which randomly picks an index in the range [0, w.length - 1] (inclusive) and returns it. The probability of picking an index i is w[i] / sum(w).
For example, if w = [1, 3], the probability of picking index 0 is 1 / (1 + 3) = 0.25 (i.e., 25%), and the probability of picking index 1 is 3 / (1 + 3) = 0.75 (i.e., 75%).

Example 1:

Input
["Solution","pickIndex"]
[[[1]],[]]
Output
[null,0]

Explanation
Solution solution = new Solution([1]);
solution.pickIndex(); // return 0. The only option is to return 0 since there is only one element in w.
Example 2:

Input
["Solution","pickIndex","pickIndex","pickIndex","pickIndex","pickIndex"]
[[[1,3]],[],[],[],[],[]]
Output
[null,1,1,1,1,0]

Explanation
Solution solution = new Solution([1, 3]);
solution.pickIndex(); // return 1. It is returning the second element (index = 1) that has a probability of 3/4.
solution.pickIndex(); // return 1
solution.pickIndex(); // return 1
solution.pickIndex(); // return 1
solution.pickIndex(); // return 0. It is returning the first element (index = 0) that has a probability of 1/4.

Since this is a randomization problem, multiple answers are allowed.
All of the following outputs can be considered correct:
[null,1,1,1,1,0]
[null,1,1,1,1,1]
[null,1,1,1,0,0]
[null,1,1,1,0,1]
[null,1,0,1,0,0]
......
and so on.

Constraints:

1 <= w.length <= 10^4
1 <= w[i] <= 10^5
pickIndex will be called at most 104 times.
'''
import bisect
import collections
import random
import time


class Solution(object):
    # O(sum(w))
    def __init__(self, w):
        """
        :type w: List[int]
        """
        start = time.time()
        # w = [1, 3]
        self.N = len(w)
        self.A = [0] * sum(w)
        cnt = 0
        for num in w:
            for _ in range(num):
                self.A[cnt] = num
                cnt += 1
        # print(self.A)
        self.build_time =  time.time() - start

    def pickIndex(self):
        """
        :rtype: int
        """
        return random.choices(self.A)

class Solution2(object):
    # O(len(w)); len(w) << sum(w)
    def __init__(self, w):
        """
        :type w: List[int]
        """
        start = time.time()
        # w = [1, 3, 4, 5] -> [1, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]
        self.N = len(w)
        self.prefix_sum = [0] * len(w)
        self.sum = sum(w)
        for i, num in enumerate(w):
            self.prefix_sum[i] += num
        # print(self.prefix_sum) # [1, 4, 8, 14]  total_sum = 13  [1, 13]
        self.prefix_sum[-1] += 1
        self.build_time = time.time() - start

    def pickIndex(self):
        """
        :rtype: int
        """
        pick_number = random.randint(1, self.sum)  # [1, 13]
        return [bisect.bisect_left(self.prefix_sum, pick_number)]



# 测试函数
def test(N=10**7, M=10**7):
    print(f"\n🎯 Testing with N={N} weights and M={M} pickIndex calls")

    # 构造随机权重数组，每个权重值 1 ~ 100
    w = [random.randint(1, 100) for _ in range(N)]

    # 测试 Solution1
    s1_start = time.time()
    sol1 = Solution(w)
    pick_start = time.time()
    for _ in range(M):
        sol1.pickIndex()
    s1_total = time.time() - s1_start
    s1_pick_time = time.time() - pick_start

    print(f"🔴 Solution1 - Build time: {sol1.build_time:.4f}s, Pick time: {s1_pick_time:.4f}s, Total: {s1_total:.4f}s")

    # 测试 Solution2
    s2_start = time.time()
    sol2 = Solution2(w)
    pick_start = time.time()
    for _ in range(M):
        sol2.pickIndex()
    s2_total = time.time() - s2_start
    s2_pick_time = time.time() - pick_start

    print(f"🟢 Solution2 - Build time: {sol2.build_time:.4f}s, Pick time: {s2_pick_time:.4f}s, Total: {s2_total:.4f}s")

test()



# # Your Solution object will be instantiated and called as such:
# w = [random.randint(10000, 100000) for i in range(100)]
# print(w)
#
# obj = Solution(w)
# for N in [100, 1000, 10000]:
#     d = collections.defaultdict(int)
#     s1 = time.time()
#     for _ in range(N):
#         param_1 = obj.pickIndex()
#         # print(param_1, end='')
#         d[param_1[0]] += 1
#     s2 = time.time()
#     print(d)
#     print("N=", N, " T=", s2-s1)
#     print('\n')
#     d = collections.defaultdict(int)
#
#
# obj2 = Solution2(w)
# for N in [100, 1000, 10000]:
#     d = collections.defaultdict(int)
#     s1 = time.time()
#     for _ in range(N):
#         param_2 = obj2.pickIndex()
#         # print(param_1, end='')
#         d[param_2[0]] += 1
#     s2 = time.time()
#     print(d)
#     print("N=", N, " T=", s2 - s1)
#     print('\n')
#     d = collections.defaultdict(int)