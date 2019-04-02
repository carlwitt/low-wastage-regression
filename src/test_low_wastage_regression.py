"""
	
"""
from unittest import TestCase
import pandas as pd

from low_wastage_regression import LowWastageRegression

__author__ = 'Carl Witt'
__email__ = 'wittcarx@informatik.hu-berlin.de'


class TestLowWastageRegression(TestCase):

	def test_transform(self):

		# negative values
		data = pd.DataFrame(dict(rss=[-1, 2, 7], run_time=1, input_size=0))
		lwr = LowWastageRegression(data, 'input_size', 'rss', 'run_time', 0.5, 1)

		lwr.__transform__(data)
		self.assertListEqual(list(data['rss']), [0, 3/8, 1])

		lwr.__inverse_transform__(data)
		self.assertListEqual(list(data['rss']), [-1, 2, 7])

		# zero-variance
		data = pd.DataFrame(dict(rss=[7, 7, 7], run_time=1, input_size=0))
		lwr = LowWastageRegression(data, 'input_size', 'rss', 'run_time', 0.5, 1)

		lwr.__transform__(data)
		self.assertListEqual(list(data['rss']), [0, 0, 0])

		lwr.__inverse_transform__(data)
		self.assertListEqual(list(data['rss']), [7, 7, 7])

	def test_perfectly_predictable(self):

		# perfectly correlated values
		data = pd.DataFrame(dict(rss=[4, 10, 6, 2, 12], run_time=1, input_size=[2, 5, 3, 1, 6]))
		lwr = LowWastageRegression(data, 'input_size', 'rss', 'run_time', 0.5, 1)
		print(lwr.model)
		print(lwr.quality)




