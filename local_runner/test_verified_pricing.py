"""Only the freshly verified night interval changes; no old test is edited."""
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path
import unittest
import trial_helpers as helpers
from budget_adapter import reservation_upper


class VerifiedPricingChecks(unittest.TestCase):
    def test_verified_window_uses_offpeak_only_within_its_bounds(self):
        facts = json.loads((Path(__file__).parent / 'PRICING_VERIFICATION.json').read_text())
        start = datetime.fromisoformat(facts['verified_window_utc']['start'])
        end = datetime.fromisoformat(facts['verified_window_utc']['end_exclusive'])
        self.assertEqual(helpers.multiplier(start - timedelta(seconds=1)), Decimal(2))
        self.assertEqual(helpers.multiplier(start), Decimal(1))
        self.assertEqual(helpers.multiplier(end - timedelta(seconds=1)), Decimal(1))
        self.assertEqual(helpers.multiplier(end), Decimal(2))

    def test_unknown_dates_and_historical_peak_policy_stay_conservative(self):
        self.assertEqual(helpers.multiplier(datetime(2026, 10, 29, 0, tzinfo=timezone.utc)), Decimal(2))
        self.assertEqual(helpers.multiplier(datetime(2026, 11, 1, 0, tzinfo=timezone.utc)), Decimal(2))
        self.assertEqual(helpers.multiplier(datetime(2026, 10, 28, 2, tzinfo=timezone.utc)), Decimal(2))

    def test_peak_reservations_and_total_cap_unchanged(self):
        self.assertEqual(helpers.HARD, Decimal(25))
        self.assertEqual(helpers.TARGET, Decimal(20))
        self.assertEqual(reservation_upper(49152), Decimal('2.490368'))
        usage = {'prompt_tokens':1000000, 'prompt_cache_hit_tokens':0, 'completion_tokens':1000000}
        self.assertEqual(helpers.cost(usage, Decimal(1)), Decimal(5))
        self.assertEqual(helpers.cost(usage, Decimal(2)), Decimal(10))


if __name__ == '__main__':
    unittest.main(verbosity=2)
