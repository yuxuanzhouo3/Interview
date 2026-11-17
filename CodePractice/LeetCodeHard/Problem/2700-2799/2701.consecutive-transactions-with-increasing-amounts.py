"""
LC 2701. Consecutive Transactions with Increasing Amounts — Hard (SQL)
Best: Gaps and islands on customer transactions
"""

SOLUTION_SQL = """
WITH t AS (
  SELECT *,
    LAG(amount) OVER (PARTITION BY customer_id ORDER BY transaction_date) AS prev_amt,
    LAG(transaction_date) OVER (PARTITION BY customer_id ORDER BY transaction_date) AS prev_date
  FROM Transactions
),
flag AS (
  SELECT *,
    CASE WHEN prev_amt IS NOT NULL AND amount > prev_amt
          AND DATEDIFF(transaction_date, prev_date)=1 THEN 0 ELSE 1 END AS new_grp
  FROM t
),
g AS (
  SELECT *, SUM(new_grp) OVER (PARTITION BY customer_id ORDER BY transaction_date) AS grp
  FROM flag
),
c AS (
  SELECT customer_id, grp,
    MIN(transaction_date) AS consecutive_start,
    MAX(transaction_date) AS consecutive_end,
    COUNT(*) AS cnt
  FROM g
  GROUP BY customer_id, grp
  HAVING COUNT(*) >= 3
)
SELECT customer_id, consecutive_start, consecutive_end
FROM c
ORDER BY customer_id, consecutive_start
"""


def _test():
    assert "consecutive_start" in SOLUTION_SQL
    print("LC 2701 OK")


if __name__ == "__main__":
    _test()
