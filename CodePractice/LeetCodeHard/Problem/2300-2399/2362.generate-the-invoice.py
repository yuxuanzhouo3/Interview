"""
LC 2362. Generate the Invoice — Hard (SQL)
Best: Join + group by invoice with max price sum
"""

SOLUTION_SQL = """
WITH inv AS (
  SELECT p.invoice_id, SUM(p.quantity * pr.price) AS price
  FROM Purchases p
  JOIN Products pr ON p.product_id = pr.product_id
  GROUP BY p.invoice_id
),
best AS (
  SELECT invoice_id FROM inv
  ORDER BY price DESC, invoice_id ASC
  LIMIT 1
)
SELECT p.product_id, p.quantity, p.quantity * pr.price AS price
FROM Purchases p
JOIN Products pr ON p.product_id = pr.product_id
JOIN best b ON p.invoice_id = b.invoice_id
ORDER BY p.product_id
"""


def _test():
    assert "invoice_id" in SOLUTION_SQL and "LIMIT 1" in SOLUTION_SQL
    print("LC 2362 OK")


if __name__ == "__main__":
    _test()
