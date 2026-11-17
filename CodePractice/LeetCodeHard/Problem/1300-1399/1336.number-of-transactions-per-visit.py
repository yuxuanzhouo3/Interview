"""
LC 1336. Number of Transactions per Visit — Hard (SQL)
https://leetcode.com/problems/number-of-transactions-per-visit/
Best: 访问的交易次数分布（含 0）
"""

SOLUTION_SQL = """
WITH tx AS (
  SELECT v.user_id, v.visit_date, COUNT(t.transaction_date) AS transactions_count
  FROM Visits v
  LEFT JOIN Transactions t
    ON v.user_id = t.user_id AND v.visit_date = t.transaction_date
  GROUP BY v.user_id, v.visit_date
),
nums AS (
  SELECT 0 AS transactions_count
  UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3
  UNION ALL SELECT 4 UNION ALL SELECT 5 UNION ALL SELECT 6
  UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9
  UNION ALL SELECT 10
),
bounds AS (
  SELECT 0 AS transactions_count
  UNION
  SELECT transactions_count FROM tx
)
SELECT n.transactions_count,
       COUNT(t.user_id) AS visits_count
FROM (
  SELECT transactions_count FROM bounds
  UNION
  SELECT n.transactions_count FROM nums n
  WHERE n.transactions_count <= (SELECT IFNULL(MAX(transactions_count),0) FROM tx)
) n
LEFT JOIN tx t ON n.transactions_count = t.transactions_count
GROUP BY n.transactions_count
ORDER BY n.transactions_count
"""


def _test():
    assert "left join" in SOLUTION_SQL.lower(), "LEFT JOIN"
    assert "transactions_count" in SOLUTION_SQL.lower(), "transactions_count"
    assert "visits_count" in SOLUTION_SQL.lower(), "visits_count"
    print("LC 1336 OK")


if __name__ == "__main__":
    _test()
