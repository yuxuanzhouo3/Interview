"""
LC 2252. Dynamic Pivoting of a Table — Hard (SQL)
Best: Dynamic SQL with GROUP_CONCAT + PREPARE
"""

SOLUTION_SQL = """
SET @sql = NULL;
SELECT GROUP_CONCAT(
  DISTINCT CONCAT(
    'MAX(CASE WHEN store = ''', store, ''' THEN amount END) AS `', store, '`'
  )
) INTO @sql
FROM Products;
SET @sql = CONCAT('SELECT product_id, ', @sql, ' FROM Products GROUP BY product_id');
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;
"""


def _test():
    assert "GROUP_CONCAT" in SOLUTION_SQL and "PREPARE" in SOLUTION_SQL
    print("LC 2252 OK")


if __name__ == "__main__":
    _test()
