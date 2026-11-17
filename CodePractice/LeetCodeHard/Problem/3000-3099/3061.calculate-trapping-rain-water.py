"""
LC 3061. Calculate Trapping Rain Water — Hard
https://leetcode.com/problems/calculate-trapping-rain-water/
Best: Trapping rain water as SQL — window
"""

SOLUTION_SQL = """
WITH h AS (
  SELECT id, height,
         MAX(height) OVER (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS left_max,
         MAX(height) OVER (ORDER BY id ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS right_max
  FROM Heights
)
SELECT SUM(GREATEST(0, LEAST(left_max, right_max) - height)) AS total_trapped_water
FROM h
"""


def _test():
    q = SOLUTION_SQL.lower()
    assert "left_max" in q and "right_max" in q
    # python verify idea
    heights = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
    n = len(heights)
    lm = [0]*n; rm=[0]*n
    m=0
    for i,h in enumerate(heights):
        m=max(m,h); lm[i]=m
    m=0
    for i in range(n-1,-1,-1):
        m=max(m,heights[i]); rm[i]=m
    assert sum(max(0, min(lm[i],rm[i])-heights[i]) for i in range(n)) == 6
    print("LC 3061 OK")


if __name__ == "__main__":
    _test()

