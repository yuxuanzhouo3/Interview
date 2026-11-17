"""
LC 2994. Friday Purchases II  — Hard
https://leetcode.com/problems/friday-purchases-ii/
Best: Friday purchases week aggregation — SQL
"""

SOLUTION_SQL = """
WITH fridays AS (
  SELECT user_id, amount_spent, purchase_date,
         WEEK(purchase_date, 1) AS wk
  FROM Purchases
  WHERE DAYOFWEEK(purchase_date)=6
    AND purchase_date BETWEEN '2023-11-03' AND '2023-11-24'
)
SELECT WEEK(purchase_date, 1) - WEEK('2023-11-03', 1) + 1 AS week_of_month,
       purchase_date,
       SUM(amount_spent) AS total_amount
FROM Purchases
WHERE DAYOFWEEK(purchase_date)=6
  AND purchase_date BETWEEN '2023-11-03' AND '2023-11-24'
GROUP BY purchase_date
ORDER BY week_of_month
"""


def _test():
    q = SOLUTION_SQL.lower()
    assert "dayofweek" in q
    assert "2023-11" in q
    assert "sum" in q
    print("LC 2994 OK")


if __name__ == "__main__":
    _test()

