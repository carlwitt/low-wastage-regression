import unittest
from functools import partial

import pandas as pd

from wastage import Wastage, wastage_exponential_prop_ttf, wastage_simple, oversizing_wastage_exponential, \
	undersizing_wastage_exponential, wastage_exponential


def wastage_exponential_naive(df: pd.DataFrame, relative_ttf: float, base: float) -> Wastage:
	"""
	Naive implementation for double checking correctness.
	Computes wastage with exponential failure handling strategy by summing up wastage for every job in a set
	:param df:
	:param relative_ttf:
	:param base:
	:return:
	"""
	assert len(df) > 0
	oversizing = 0
	undersizing = 0
	failures = 0

	for row in df.itertuples():
		oversizing += oversizing_wastage_exponential(real_usage=row.rss, run_time=row.run_time,
													 first_allocation=row.first_allocation, base=base)
		u, fails = undersizing_wastage_exponential(real_usage=row.rss, abs_ttf=row.run_time * relative_ttf,
												   first_allocation=row.first_allocation, base=base)
		undersizing += u
		failures += fails

	return Wastage(oversizing=oversizing, undersizing=undersizing, usage=sum(df['run_time'] * df['rss']),
				   failures=failures)


class TestWastage(unittest.TestCase):


	def test_oversizing_exponential(self):

		# succeeds on first attempt
		oversizing = oversizing_wastage_exponential(real_usage=3, run_time=2.0, first_allocation=7.0, base=2)
		self.assertAlmostEqual(oversizing, 4.0 * 2)

		# succeeds on second attempt
		oversizing = oversizing_wastage_exponential(real_usage=10, run_time=2.0, first_allocation=7.0, base=2)
		self.assertAlmostEqual(oversizing, 4.0 * 2)

		# succeeds on third attempt
		oversizing = oversizing_wastage_exponential(real_usage=20, run_time=2.0, first_allocation=7.0, base=2)
		self.assertAlmostEqual(oversizing, 8.0 * 2)

	def test_undersizing_exponential(self):

		# succeeds on first attempt
		undersizing = undersizing_wastage_exponential(real_usage=3, abs_ttf=2, first_allocation=5, base=2)
		self.assertAlmostEqual(undersizing[0], 0, 6)

		# succeeds on second attempt
		undersizing = undersizing_wastage_exponential(real_usage=10, abs_ttf=2, first_allocation=5, base=2)
		self.assertAlmostEqual(undersizing[0], 5 * 2, 6)

		# succeeds on fifth attempt
		# 1 + 2 + 4 + 8 = 15
		undersizing = undersizing_wastage_exponential(real_usage=10, abs_ttf=2, first_allocation=1, base=2)
		self.assertAlmostEqual(undersizing[0], 15 * 2, 6)

	def test_vectorized(self):
		"""
		Make sure the vectorized wastage calculation return the same results as the naive implementation.
		Interestingly, itertuples is much faster than iterrows, reducing runtimes from 3-4 seconds to 0.15 seconds
		:return:
		"""
		import pandas as pd
		from time import time as now

		rel_ttf = 0.01

		df = pd.read_csv("../../sample_data/sample.csv.gz")
		df['first_allocation'] = 3

		am = df['rss'].max() / 2
		av = df['rss'].max()

		for fast, slow in [
			(partial(wastage_exponential, base=2), partial(wastage_exponential_naive, base=2)),
		]:
			before = now()
			fast_wastage = fast(df, relative_ttf=rel_ttf)
			print("Fast version: {} sec".format(now() - before))

			before = now()
			slow_wastage = slow(df, relative_ttf=rel_ttf)
			print("Slow version: {} sec".format(now() - before))

			self.assertEqual(fast_wastage.failures, slow_wastage.failures)
			self.assertAlmostEqual(fast_wastage.oversizing, slow_wastage.oversizing, 6)
			self.assertAlmostEqual(fast_wastage.undersizing, slow_wastage.undersizing, 6)
			self.assertAlmostEqual(fast_wastage.maq, slow_wastage.maq, 6)
			self.assertTrue(fast_wastage.maq <= 1)

	def test_wastage_simple(self):

		import pandas as pd
		from time import time as now

		df = pd.DataFrame(dict(
			peak_mem=[1, 2, 3, 4, 5],
		))
		df['first_allocation'] = 3

		w = wastage_simple(df, resource_column='peak_mem')
		self.assertEqual(w.failures, 2)
		self.assertEqual(w.usage, 6)
		self.assertEqual(w.oversizing, 3)
		self.assertEqual(w.undersizing, 6)

	def test_wastage_proportional_ttf(self):

		import pandas as pd

		df = pd.DataFrame(dict(
			rss=[10],
			run_time=[5]
		))
		df['first_allocation'] = 1

		w = wastage_exponential_prop_ttf(df, 2)
		self.assertEqual(w.failures, 4)
		self.assertEqual(w.usage, 50)
		self.assertEqual(w.oversizing, 30)
		self.assertEqual(w.undersizing, 42.5)
