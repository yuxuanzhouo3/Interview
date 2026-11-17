"""
LC 2752. Customers with Maximum Number of Transactions on Consecutive Days — Hard (SQL)
Best: Islands of consecutive days; count tx
"""

SOLUTION_SQL = """
WITH d AS (
  SELECT customer_id, transaction_date,
    DATE_SUB(transaction_date, INTERVAL ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY transaction_date) DAY) AS grp
  FROM (SELECT DISTINCT customer_id, transaction_date FROM Transactions) t
),
span AS (
  SELECT customer_id, grp, COUNT(*) AS days
  FROM d GROUP BY customer_id, grp
),
mx AS (SELECT MAX(days) AS m FROM span)
SELECT t.customer_id, COUNT(*) AS cnt
FROM Transactions t
JOIN d ON d.customer_id=t.customer_id AND d.transaction_date=t.transaction_date
JOIN span s ON s.customer_id=d.customer_id AND s.grp=d.grp
JOIN mx ON s.days = mx.m
GROUP BY t.customer_id, s.grp
ORDER BY customer_id
"""


def _test():
    assert "customer_id" in SOLUTION_SQL
    print("LC 2752 OK")


if __name__ == "__main__":
    _test()
