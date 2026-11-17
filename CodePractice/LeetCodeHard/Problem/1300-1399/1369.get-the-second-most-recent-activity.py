"""
LC 1369. Get the Second Most Recent Activity — Hard (SQL)
https://leetcode.com/problems/get-the-second-most-recent-activity/
Best: 第二近活动，仅一次则取该次
"""

SOLUTION_SQL = """
SELECT username, activity, startDate, endDate
FROM (
  SELECT *,
         COUNT(*) OVER (PARTITION BY username) AS cnt,
         RANK() OVER (PARTITION BY username ORDER BY startDate DESC) AS rk
  FROM UserActivity
) t
WHERE rk = 2 OR cnt = 1
"""


def _test():
    assert "rank()" in SOLUTION_SQL.lower(), "RANK()"
    assert "cnt = 1" in SOLUTION_SQL.lower(), "cnt = 1"
    assert "rk = 2" in SOLUTION_SQL.lower(), "rk = 2"
    print("LC 1369 OK")


if __name__ == "__main__":
    _test()
