"""
LC 2254. Design Video Sharing Platform — Hard
https://leetcode.com/problems/design-video-sharing-platform/
Tags: Design, Hash, Heap
Best: 最小可用 id + 哈希存内容/观看/点赞
"""

from typing import List
import heapq


class VideoSharingPlatform:
    def __init__(self):
        self.next_id = 0
        self.reuse = []
        self.videos = {}  # id -> content
        self.views = {}
        self.likes = {}
        self.dislikes = {}

    def _alloc(self) -> int:
        if self.reuse:
            return heapq.heappop(self.reuse)
        vid = self.next_id
        self.next_id += 1
        return vid

    def upload(self, video: str) -> int:
        vid = self._alloc()
        self.videos[vid] = video
        self.views[vid] = 0
        self.likes[vid] = 0
        self.dislikes[vid] = 0
        return vid

    def remove(self, videoId: int) -> None:
        if videoId in self.videos:
            del self.videos[videoId]
            del self.views[videoId]
            del self.likes[videoId]
            del self.dislikes[videoId]
            heapq.heappush(self.reuse, videoId)

    def watch(self, videoId: int, startMinute: int, endMinute: int) -> str:
        if videoId not in self.videos:
            return "-1"
        self.views[videoId] += 1
        content = self.videos[videoId]
        return content[startMinute:min(endMinute, len(content) - 1) + 1]

    def like(self, videoId: int) -> None:
        if videoId in self.videos:
            self.likes[videoId] += 1

    def dislike(self, videoId: int) -> None:
        if videoId in self.videos:
            self.dislikes[videoId] += 1

    def getLikesAndDislikes(self, videoId: int) -> List[int]:
        if videoId not in self.videos:
            return [-1]
        return [self.likes[videoId], self.dislikes[videoId]]

    def getViews(self, videoId: int) -> int:
        if videoId not in self.videos:
            return -1
        return self.views[videoId]


def _test():
    p = VideoSharingPlatform()
    assert p.upload("abc") == 0
    assert p.upload("def") == 1
    assert p.watch(0, 0, 1) == "ab"
    p.like(0)
    assert p.getLikesAndDislikes(0) == [1, 0]
    assert p.getViews(0) == 1
    p.remove(0)
    assert p.upload("xyz") == 0
    print("LC 2254 OK")


if __name__ == "__main__":
    _test()

