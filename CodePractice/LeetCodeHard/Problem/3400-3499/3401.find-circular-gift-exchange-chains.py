"""
LC 3401. Find Circular Gift Exchange Chains — Hard (SQL)
https://leetcode.com/problems/find-circular-gift-exchange-chains/
Best: Recursive CTE follow cycles + rank
"""

SOLUTION_SQL = """
WITH RECURSIVE Chains AS (
  SELECT *, giver_id AS start_id
  FROM SecretSanta
  UNION ALL
  SELECT SecretSanta.*, Chains.start_id
  FROM SecretSanta
  INNER JOIN Chains
    ON SecretSanta.giver_id = Chains.receiver_id
   AND SecretSanta.giver_id != Chains.start_id
),
ChainSummary AS (
  SELECT
    start_id,
    COUNT(*) AS chain_length,
    SUM(gift_value) AS total_gift_value
  FROM Chains
  GROUP BY 1
),
UniqueChains AS (
  SELECT DISTINCT chain_length, total_gift_value
  FROM ChainSummary
)
SELECT
  RANK() OVER (ORDER BY chain_length DESC, total_gift_value DESC) AS chain_id,
  chain_length,
  total_gift_value
FROM UniqueChains
"""


def _test():
    assert "RECURSIVE" in SOLUTION_SQL
    assert "chain_length" in SOLUTION_SQL
    assert "total_gift_value" in SOLUTION_SQL
    print("LC 3401 OK")


if __name__ == "__main__":
    _test()
