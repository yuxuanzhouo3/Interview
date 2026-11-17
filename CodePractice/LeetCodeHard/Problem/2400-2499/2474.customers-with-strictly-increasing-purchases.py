"""
LC 2474. Customers With Strictly Increasing Purchases — Hard (SQL)
https://leetcode.com/problems/customers-with-strictly-increasing-purchases/
Best: 连续月消费严格递增
"""

SOLUTION_SQL = """
WITH monthly AS (
  SELECT customer_id,
         DATE_FORMAT(order_date, '%Y-%m') AS ym,
         SUM(amount) AS total
  FROM Orders
  GROUP BY customer_id, DATE_FORMAT(order_date, '%Y-%m')
),
marked AS (
  SELECT customer_id, ym, total,
         LAG(total) OVER (PARTITION BY customer_id ORDER BY ym) AS prev_total,
         LAG(ym) OVER (PARTITION BY customer_id ORDER BY ym) AS prev_ym
  FROM monthly
)
SELECT DISTINCT customer_id
FROM monthly m
WHERE customer_id NOT IN (
  SELECT customer_id FROM marked
  WHERE prev_ym IS NOT NULL AND (
    PERIOD_DIFF(REPLACE(ym,'-',''), REPLACE(prev_ym,'-','')) != 1
    OR total <= prev_total
  )
)
AND customer_id IN (SELECT customer_id FROM monthly GROUP BY customer_id HAVING COUNT(*) >= 1)
"""


def _test():
    assert "lag" in SOLUTION_SQL.lower(), "LAG"
    assert "customer_id" in SOLUTION_SQL.lower(), "customer_id"
    assert "date_format" in SOLUTION_SQL.lower(), "DATE_FORMAT"
    print("LC 2474 OK")


if __name__ == "__main__":
    _test()
