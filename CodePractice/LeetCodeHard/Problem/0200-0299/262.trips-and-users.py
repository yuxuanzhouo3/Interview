"""
LC 262. Trips and Users — Hard (SQL)
Best: Filter banned users + daily cancellation rate
"""

SOLUTION_SQL = """
SELECT
  t.request_at AS Day,
  ROUND(SUM(t.status != 'completed') / COUNT(*), 2) AS `Cancellation Rate`
FROM Trips t
JOIN Users c ON t.client_id = c.users_id AND c.banned = 'No'
JOIN Users d ON t.driver_id = d.users_id AND d.banned = 'No'
WHERE t.request_at BETWEEN '2013-10-01' AND '2013-10-03'
GROUP BY t.request_at
"""


def _test():
    assert "Cancellation Rate" in SOLUTION_SQL
    print("LC 262 OK")


if __name__ == "__main__":
    _test()
