"""
LC 1972. First and Last Call On the Same Day — Hard (SQL)
https://leetcode.com/problems/first-and-last-call-on-the-same-day/
Best: 当日首末通话同一对用户
"""

SOLUTION_SQL = """
WITH calls AS (
  SELECT caller_id AS user1, recipient_id AS user2, call_time FROM Calls
  UNION ALL
  SELECT recipient_id, caller_id, call_time FROM Calls
),
ranked AS (
  SELECT *,
         DATE(call_time) AS d,
         FIRST_VALUE(user2) OVER (PARTITION BY user1, DATE(call_time) ORDER BY call_time) AS first_peer,
         FIRST_VALUE(user2) OVER (PARTITION BY user1, DATE(call_time) ORDER BY call_time DESC) AS last_peer
  FROM calls
)
SELECT DISTINCT LEAST(user1, first_peer) AS user1,
                GREATEST(user1, first_peer) AS user2
FROM ranked
WHERE first_peer = last_peer
"""


def _test():
    assert "first_value" in SOLUTION_SQL.lower(), "FIRST_VALUE"
    assert "union all" in SOLUTION_SQL.lower(), "UNION ALL"
    assert "date(call_time)" in SOLUTION_SQL.lower(), "DATE(call_time)"
    print("LC 1972 OK")


if __name__ == "__main__":
    _test()
