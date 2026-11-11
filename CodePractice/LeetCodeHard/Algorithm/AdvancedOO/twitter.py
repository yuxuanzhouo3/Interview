"""
8. Design Twitter
Template: query(newsFeed) / update(follow) / add(post/follow) / del(unfollow)
App: LC 355。
"""

from collections import defaultdict
import heapq
import itertools


class Twitter:
    def __init__(self):
        self.time = itertools.count(step=-1)  # 越大越新；用负数入堆
        self.tweets = defaultdict(list)  # user -> [(time, tweetId)]
        self.followees = defaultdict(set)

    def add(self, userId, tweetId=None, followeeId=None):
        if tweetId is not None:
            self.postTweet(userId, tweetId)
        if followeeId is not None:
            self.follow(userId, followeeId)

    def postTweet(self, userId, tweetId):
        self.tweets[userId].append((next(self.time), tweetId))

    def query(self, userId):
        return self.getNewsFeed(userId)

    def getNewsFeed(self, userId):
        people = self.followees[userId] | {userId}
        heap = []
        for u in people:
            if self.tweets[u]:
                t, tid = self.tweets[u][-1]
                heapq.heappush(heap, (t, tid, u, len(self.tweets[u]) - 1))
        feed = []
        while heap and len(feed) < 10:
            t, tid, u, i = heapq.heappop(heap)
            feed.append(tid)
            if i > 0:
                t2, tid2 = self.tweets[u][i - 1]
                heapq.heappush(heap, (t2, tid2, u, i - 1))
        return feed

    def update(self, followerId, followeeId):
        self.follow(followerId, followeeId)
        return True

    def follow(self, followerId, followeeId):
        if followerId != followeeId:
            self.followees[followerId].add(followeeId)

    def delete(self, followerId, followeeId):
        self.unfollow(followerId, followeeId)

    def unfollow(self, followerId, followeeId):
        self.followees[followerId].discard(followeeId)


def _test():
    tw = Twitter()
    tw.postTweet(1, 5)
    assert tw.getNewsFeed(1) == [5]
    tw.follow(1, 2)
    tw.postTweet(2, 6)
    assert tw.getNewsFeed(1) == [6, 5]
    tw.unfollow(1, 2)
    assert tw.getNewsFeed(1) == [5]
    print("Twitter OK")


APPLICATION = """
- 关注集合 + 各用户推文列表；新闻流用堆做 k 路归并。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
