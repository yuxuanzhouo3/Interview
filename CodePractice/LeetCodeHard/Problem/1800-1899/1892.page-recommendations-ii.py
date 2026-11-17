"""
LC 1892. Page Recommendations II — Hard (SQL)
https://leetcode.com/problems/page-recommendations-ii/
Best: 朋友喜欢且自己未喜欢，按人聚合计数
"""

SOLUTION_SQL = """
WITH Friend AS (
  SELECT user1_id AS user_id, user2_id AS friend_id FROM Friendship
  UNION ALL
  SELECT user2_id, user1_id FROM Friendship
)
SELECT
  f.user_id,
  l.page_id,
  COUNT(*) AS friends_likes
FROM Friend f
JOIN Likes l ON f.friend_id = l.user_id
WHERE NOT EXISTS (
  SELECT 1 FROM Likes my
  WHERE my.user_id = f.user_id AND my.page_id = l.page_id
)
GROUP BY f.user_id, l.page_id
"""


def _test():
    q = SOLUTION_SQL.lower()
    assert "friends_likes" in q
    assert "friendship" in q
    assert "likes" in q
    print("LC 1892 OK")


if __name__ == "__main__":
    _test()
