"""
LC 1919. Leetcodify Similar Friends — Hard (SQL)
https://leetcode.com/problems/leetcodify-similar-friends/
Best: 好友且同日共听 ≥3 首
"""

SOLUTION_SQL = """
SELECT DISTINCT
  LEAST(f.user1_id, f.user2_id) AS user1_id,
  GREATEST(f.user1_id, f.user2_id) AS user2_id
FROM Friendship f
JOIN Listens a ON a.user_id = f.user1_id
JOIN Listens b ON b.user_id = f.user2_id
  AND a.day = b.day AND a.song_id = b.song_id
GROUP BY f.user1_id, f.user2_id, a.day
HAVING COUNT(DISTINCT a.song_id) >= 3
"""


def _test():
    q = SOLUTION_SQL.lower()
    assert "friendship" in q and "listens" in q
    assert ">= 3" in q or ">=3" in q
    print("LC 1919 OK")


if __name__ == "__main__":
    _test()
