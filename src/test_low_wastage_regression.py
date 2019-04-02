"""
	
"""
from unittest import TestCase
import pandas as pd

from low_wastage_regression import LowWastageRegression
from wastage import Wastage

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
		self.assertAlmostEqual(lwr.quality.maq, 1, 2)

	def test_train_evaluation(self):
		"""
		TODO test that wastage (MAQ/failures,etc.) are identical when training on a data set and then obtaining predictions on that data set.
		"""
		pass

		df = pd.read_csv("../../sample_data/sample.csv.gz")

		training = df.sample(frac=0.1, random_state=13)
		lwr = LowWastageRegression(training, predictor_column='input_size',
								   resource_column='rss',
								   run_time_column='run_time',
								   relative_time_to_failure=0.5,
								   min_allocation=0.01)

		print("best_params: {0}".format(lwr.model))
		print("maq: {:.2f}% failures: {:.2f}%".format(lwr.quality.maq * 100, lwr.quality.failures / df.size * 100))

		evaluation = training
		evaluation['first_allocation'] = lwr.predict(evaluation)
		w = Wastage.exponential(evaluation, 0.5, resource_column='rss', first_allocation_column='first_allocation',
								run_time_column='run_time')
		self.assertAlmostEqual(lwr.quality.oversizing, w.oversizing, 2)
		self.assertAlmostEqual(lwr.quality.undersizing, w.undersizing, 2)
		self.assertAlmostEqual(lwr.quality.failures, w.failures, 2)
		self.assertAlmostEqual(lwr.quality.maq, w.maq, 2)



