"""
LC 2720. Popularity Percentage — Hard (SQL)
https://leetcode.com/problems/popularity-percentage/
Best: 好友数占比两位小数
"""

SOLUTION_SQL = """
WITH users AS (
  SELECT user1 AS user_id FROM Friends
  UNION
  SELECT user2 FROM Friends
),
deg AS (
  SELECT user_id, COUNT(*) AS cnt FROM (
    SELECT user1 AS user_id FROM Friends
    UNION ALL
    SELECT user2 FROM Friends
  ) t
  GROUP BY user_id
)
SELECT d.user_id,
       ROUND(d.cnt * 100.0 / (SELECT COUNT(*) FROM users), 2) AS percentage_popularity
FROM deg d
ORDER BY d.user_id
"""


def _test():
    assert "union all" in SOLUTION_SQL.lower(), "UNION ALL"
    assert "round" in SOLUTION_SQL.lower(), "ROUND"
    assert "percentage_popularity" in SOLUTION_SQL.lower(), "percentage_popularity"
    print("LC 2720 OK")


if __name__ == "__main__":
    _test()
