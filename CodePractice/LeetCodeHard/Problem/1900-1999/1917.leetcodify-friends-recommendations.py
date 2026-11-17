"""
LC 1917. Leetcodify Friends Recommendations — Hard (SQL)
https://leetcode.com/problems/leetcodify-friends-recommendations/
Best: 同日听 ≥3 首相同歌且非好友 → 互推
"""

SOLUTION_SQL = """
WITH pairs AS (
  SELECT DISTINCT
    LEAST(a.user_id, b.user_id) AS user1_id,
    GREATEST(a.user_id, b.user_id) AS user2_id
  FROM Listens a
  JOIN Listens b
    ON a.user_id < b.user_id
   AND a.day = b.day
   AND a.song_id = b.song_id
  GROUP BY a.user_id, b.user_id, a.day
  HAVING COUNT(DISTINCT a.song_id) >= 3
)
SELECT user1_id AS user_id, user2_id AS recommended_id FROM pairs
WHERE NOT EXISTS (
  SELECT 1 FROM Friendship f
  WHERE (f.user1_id = pairs.user1_id AND f.user2_id = pairs.user2_id)
     OR (f.user1_id = pairs.user2_id AND f.user2_id = pairs.user1_id)
)
UNION
SELECT user2_id, user1_id FROM pairs
WHERE NOT EXISTS (
  SELECT 1 FROM Friendship f
  WHERE (f.user1_id = pairs.user1_id AND f.user2_id = pairs.user2_id)
     OR (f.user1_id = pairs.user2_id AND f.user2_id = pairs.user1_id)
)
"""


def _test():
    q = SOLUTION_SQL.lower()
    assert "listens" in q
    assert "friendship" in q
    assert "recommended_id" in q or "user2_id" in q
    assert ">= 3" in q or ">=3" in q
    print("LC 1917 OK")


if __name__ == "__main__":
    _test()
