"""Tests for reconcile.py. Run: python3 -m unittest discover -s checks"""
import unittest
from pathlib import Path

import pandas as pd

import reconcile as rc

BOOK = Path(__file__).resolve().parent.parent / "harbor_ridge_book_3552.xlsx"


def book(rows):
    """Build a small index from dicts; fills the columns the checks expect."""
    base = {"portal": "P", "client": "C", "investing_entity": "E", "manager": "M", "fund": "F",
            "file_name": "doc", "document_type": "Other", "commitment": None, "contributed": None,
            "total_value": None, "cumulative_distributions": None, "cash_amount": None}
    d = pd.DataFrame([{**base, "row_id": i + 1, **r} for i, r in enumerate(rows)])
    for c in ("effective_date", "posted_date"):
        d[c] = pd.to_datetime(d[c])
    for c in ("commitment", "contributed", "total_value", "cumulative_distributions", "cash_amount"):
        d[c] = d[c].astype(float)
    d["position"] = list(zip(*(d[c] for c in rc.POS)))
    return d


def stmt(as_of, contributed, dist=0, value=1000, commitment=100000, **kw):
    return {"document_type": "Account Statement", "effective_date": as_of, "posted_date": as_of,
            "commitment": commitment, "contributed": contributed, "total_value": value,
            "cumulative_distributions": dist, **kw}


def call(due, amount, **kw):
    return {"document_type": "Capital Call", "effective_date": due, "posted_date": due,
            "cash_amount": -amount, **kw}


def dist(on, amount, **kw):
    return {"document_type": "Distribution Notice", "effective_date": on, "posted_date": on,
            "cash_amount": amount, **kw}


class UnitTests(unittest.TestCase):
    def test_contributed_ties(self):
        d = book([call("2024-01-10", 500), call("2024-02-10", 500), stmt("2024-03-31", 1000)])
        self.assertEqual(rc.check_contributed(d, set()), [])

    def test_contributed_counts_by_due_date_and_explains_timing(self):
        d = book([call("2024-01-10", 500), stmt("2024-03-31", 1000), call("2024-04-10", 500)])
        [f] = rc.check_contributed(d, set())
        self.assertEqual(f.rows, [2])
        self.assertEqual(f.related, [3])
        self.assertIn("after 2024-03-31", f.issue)

    def test_contributed_within_tolerance(self):
        d = book([call("2024-01-10", 999.6), stmt("2024-03-31", 1000)])
        self.assertEqual(rc.check_contributed(d, set()), [])

    def test_distributions_not_netted_from_contributed(self):
        d = book([call("2024-01-10", 1000), dist("2024-02-10", 300), stmt("2024-03-31", 1000, dist=300)])
        self.assertEqual(rc.check_contributed(d, set()) + rc.check_distributions(d, set()), [])

    def test_cash_duplicate_is_blocking_and_dropped_once(self):
        d = book([call("2024-01-10", 500, file_name="a"), call("2024-01-10", 500, file_name="b"),
                  stmt("2024-03-31", 500)])
        dups = rc.check_duplicates(d)
        self.assertEqual([(f.rows, f.severity) for f in dups], [([1, 2], rc.BLOCKING)])
        drop = rc.duplicate_rows_to_drop(dups)
        self.assertEqual(drop, {2})
        self.assertEqual(rc.check_contributed(d, drop), [])

    def test_same_file_in_different_entities_is_not_a_duplicate(self):
        d = book([{"investing_entity": "A", "file_name": "Ack_3", "effective_date": "2022-03-01",
                   "posted_date": "2022-03-01", "document_type": "Legal"},
                  {"investing_entity": "B", "file_name": "Ack_3", "effective_date": "2022-03-01",
                   "posted_date": "2022-03-01", "document_type": "Legal"}])
        self.assertEqual(rc.check_duplicates(d), [])

    def test_normalize_name(self):
        self.assertEqual(rc.normalize_name("X L.P. - Notice - X (2) (3)"), rc.normalize_name("X L.P. - Notice"))
        self.assertNotEqual(rc.normalize_name("Report_Q124"), rc.normalize_name("Report_Q224"))

    def test_value_dip_and_recovery_flags_the_dip(self):
        d = book([stmt("2024-03-31", 0, value=500000), stmt("2024-06-30", 0, value=50000),
                  stmt("2024-09-30", 0, value=520000)])
        [f] = rc.check_sanity(d)
        self.assertEqual(f.rows, [2])

    def test_paid_in_falls(self):
        d = book([stmt("2024-03-31", 1000), stmt("2024-06-30", 900)])
        [f] = rc.check_sanity(d)
        self.assertIn("fell", f.issue)

    def test_pro_rata(self):
        rows = [stmt("2024-03-31", 0, investing_entity="A", commitment=100000),
                stmt("2024-03-31", 0, investing_entity="B", commitment=200000),
                call("2024-01-10", 10000, investing_entity="A"),
                call("2024-01-10", 20000, investing_entity="B")]
        self.assertEqual(rc.check_pro_rata(book(rows), set()), [])
        rows[3] = call("2024-01-10", 30000, investing_entity="B")
        self.assertEqual(len(rc.check_pro_rata(book(rows), set())), 2)

    def test_pro_rata_missing_call_for_one_co_investor(self):
        rows = [stmt("2024-03-31", 0, investing_entity="A"), stmt("2024-03-31", 0, investing_entity="B"),
                call("2024-01-10", 10000, investing_entity="A")]
        [f] = rc.check_pro_rata(book(rows), set())
        self.assertEqual(f.position[1], "B")

    def test_file_dates(self):
        dates, _ = rc.file_dates("Capital Call Due Feb 25, 2023 - X")
        self.assertIn(pd.Timestamp("2023-02-25"), dates)
        dates, _ = rc.file_dates("Notice (24.02.23)")
        self.assertIn(pd.Timestamp("2023-02-24"), dates)
        _, quarters = rc.file_dates("Quarterly Report_Q323_1")
        self.assertEqual(quarters, [pd.Timestamp("2023-09-30")])

    def test_tax_estimate_exempt_from_posted_before_date(self):
        d = book([{"document_type": "Tax", "file_name": "2023 Tax Estimate", "effective_date": "2023-12-31",
                   "posted_date": "2023-12-06"}])
        self.assertEqual(rc.check_dates(d), [])

    def test_money_format(self):
        self.assertEqual(rc._money(-200000), "-$200,000")
        self.assertEqual(rc._money(2556.23), "$2,556.23")


class BookTests(unittest.TestCase):
    """Pins the findings on the Harbor Ridge book that the deck cites."""

    @classmethod
    def setUpClass(cls):
        cls.d = rc.load(BOOK)
        cls.f = rc.run(cls.d)
        cls.g = rc.grid(cls.d, cls.f)

    def rows(self, check, severity=None):
        return {r for f in self.f if f.check == check and severity in (None, f.severity) for r in f.rows}

    def test_shape(self):
        self.assertEqual(len(self.d), 338)
        self.assertEqual(len(self.g), 12)

    def test_check1(self):
        self.assertEqual(self.rows(1), {3, 27, 51, 76, 146, 147, 148, 195, 237, 256, 288, 291, 337})

    def test_check1_stonebridge_explained_by_later_calls(self):
        f = next(f for f in self.f if f.check == 1 and f.rows == [3])
        self.assertEqual(f.related, [6, 7])

    def test_check2(self):
        self.assertTrue({3, 27, 51, 192, 194, 256, 309} <= self.rows(2))
        self.assertTrue(set(range(86, 100)) >= self.rows(2) - {3, 27, 51, 192, 194, 256, 309})

    def test_check3(self):
        self.assertEqual(self.rows(3), {147, 288, 289, 292, 308, 313})

    def test_check4(self):
        self.assertEqual(self.rows(4, rc.BLOCKING), {101, 103})
        self.assertTrue({250, 251, 166, 167, 170, 177, 178, 179} <= self.rows(4, rc.HYGIENE))
        self.assertFalse({21, 44, 45, 68} & self.rows(4))

    def test_check5(self):
        self.assertEqual(self.rows(5), {58, 187, 188, 189, 338})

    def test_check6(self):
        self.assertTrue({16, 17, 40, 41, 64, 65, 157, 182, 224, 239} <= self.rows(6))
        self.assertFalse(set(range(134, 140)) & self.rows(6))  # tax estimates exempt

    def test_check7(self):
        self.assertTrue({337, 338, 146} <= self.rows(7))

    def test_check8_northwell_only(self):
        self.assertEqual(self.rows(8), {197, 198, 199, 200, 314, 315, 316, 317})
        stonebridge = [f for f in self.f if f.check == 8 and f.position[2].startswith("Stonebridge")]
        self.assertEqual(stonebridge, [])

    def test_two_of_twelve_reconcile(self):
        ok = {p[2] for p in rc.reconciled_positions(self.g)}
        self.assertEqual(ok, {"Ashbourne Middle Market Fund II, L.P.", "Falcon Ridge Ventures Fund III, L.P."})

    def test_eleven_stale(self):
        self.assertEqual(len(rc.staleness(self.d)), 11)

    def test_deterministic(self):
        again = rc.run(rc.load(BOOK))
        self.assertEqual([(f.check, f.rows, f.issue) for f in again], [(f.check, f.rows, f.issue) for f in self.f])


if __name__ == "__main__":
    unittest.main()
