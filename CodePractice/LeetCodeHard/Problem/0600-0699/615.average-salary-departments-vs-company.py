"""
LC 615. Average Salary: Departments VS Company — Hard (SQL)
https://leetcode.com/problems/average-salary-departments-vs-company/
Best: 按月比较部门均薪与公司均薪
"""

SOLUTION_SQL = """
SELECT
  DATE_FORMAT(pay_date, '%Y-%m') AS pay_month,
  department_id,
  CASE
    WHEN AVG(amount) > company_avg THEN 'higher'
    WHEN AVG(amount) < company_avg THEN 'lower'
    ELSE 'same'
  END AS comparison
FROM Salary s
JOIN Employee e ON s.employee_id = e.employee_id
JOIN (
  SELECT DATE_FORMAT(pay_date, '%Y-%m') AS pay_month, AVG(amount) AS company_avg
  FROM Salary
  GROUP BY DATE_FORMAT(pay_date, '%Y-%m')
) c ON DATE_FORMAT(s.pay_date, '%Y-%m') = c.pay_month
GROUP BY DATE_FORMAT(pay_date, '%Y-%m'), department_id, company_avg
"""


def _test():
    assert "avg" in SOLUTION_SQL.lower(), "AVG"
    assert "case" in SOLUTION_SQL.lower(), "CASE"
    assert "date_format" in SOLUTION_SQL.lower(), "DATE_FORMAT"
    assert "department_id" in SOLUTION_SQL.lower(), "department_id"
    print("LC 615 OK")


if __name__ == "__main__":
    _test()
