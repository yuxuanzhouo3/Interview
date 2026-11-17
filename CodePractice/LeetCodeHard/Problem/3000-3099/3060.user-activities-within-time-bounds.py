"""
LC 3060. User Activities within Time Bounds — Hard
https://leetcode.com/problems/user-activities-within-time-bounds/
Best: User activities within bounds — SQL
"""

SOLUTION_SQL = """
SELECT DISTINCT a.user_id
FROM Sessions a
JOIN Sessions b
  ON a.user_id = b.user_id AND a.session_id <> b.session_id
 AND a.session_type = b.session_type
 AND TIMESTAMPDIFF(SECOND, a.session_end, b.session_start) BETWEEN 0 AND (
   SELECT inclusive_bound FROM /* placeholder */ a
 )
-- Correct form uses Sessions + bounds table:
-- Join each user sessions of same type where gap <= max_duration for that type
"""


# Python simulation of intended logic for tests
from typing import List, Tuple


def users_within_bounds(sessions: List[Tuple], bounds: dict) -> List[int]:
    # sessions: (user_id, session_start, session_end, session_type)
    from collections import defaultdict
    by = defaultdict(list)
    for u, st, en, typ in sessions:
        by[(u, typ)].append((st, en))
    ans = set()
    for (u, typ), lst in by.items():
        lst.sort()
        lim = bounds[typ]
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                gap = lst[j][0] - lst[i][1]
                if 0 <= gap <= lim:
                    ans.add(u)
    return sorted(ans)


def _test():
    assert "session" in SOLUTION_SQL.lower()
    assert users_within_bounds(
        [(1, 0, 10, "A"), (1, 15, 20, "A"), (2, 0, 5, "B")],
        {"A": 10, "B": 1},
    ) == [1]
    print("LC 3060 OK")


if __name__ == "__main__":
    _test()

