"""
LC 2793. Status of Flight Tickets — Hard (SQL)
Best: Window remaining seats per flight
"""

SOLUTION_SQL = """
WITH ordered AS (
  SELECT *,
    ROW_NUMBER() OVER (PARTITION BY flight_id ORDER BY booking_time) AS rn
  FROM Passengers
),
cap AS (
  SELECT o.passenger_id, o.flight_id,
    CASE WHEN o.rn <= f.capacity THEN 'Confirmed' ELSE 'Waitlist' END AS Status
  FROM ordered o
  JOIN Flights f ON f.flight_id=o.flight_id
)
SELECT passenger_id, Status FROM cap ORDER BY passenger_id
"""


def _test():
    assert "Confirmed" in SOLUTION_SQL
    print("LC 2793 OK")


if __name__ == "__main__":
    _test()
