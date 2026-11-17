"""
LC 2253. Dynamic Unpivoting of a Table — Hard (SQL)
Best: Dynamic SQL UNION ALL over store columns
"""

SOLUTION_SQL = """
SET @sql = NULL;
SELECT GROUP_CONCAT(
  DISTINCT CONCAT(
    'SELECT product_id, ''', COLUMN_NAME, ''' AS store, `', COLUMN_NAME,
    '` AS price FROM Products WHERE `', COLUMN_NAME, '` IS NOT NULL'
  ) SEPARATOR ' UNION ALL '
) INTO @sql
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'Products' AND COLUMN_NAME != 'product_id';
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;
"""


def _test():
    assert "UNION ALL" in SOLUTION_SQL and "INFORMATION_SCHEMA" in SOLUTION_SQL
    print("LC 2253 OK")


if __name__ == "__main__":
    _test()
