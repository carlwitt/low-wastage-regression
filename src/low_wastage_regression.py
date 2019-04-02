"""
	Compute wastage-minimizing first allocations for jobs given historical data about their resource usage.
	Create one object per resource type (e.g., main memory, storage, etc.)
"""
from functools import partial
from typing import Callable, List, Optional
import pandas as pd
import numpy as np
import scipy.optimize as spo
import scipy.stats as sps
import statsmodels.formula.api as smf

from wastage import Wastage

__author__ = 'Carl Witt'
__email__ = 'wittcarx@informatik.hu-berlin.de'


class LowWastageRegression:

	""" If the multiplier for failed attempts is optimized, limit it to this value, e.g., increase allocation by at least 50% upon each failure."""
	min_base = 1.5

	def __init__(self, training_data: pd.DataFrame, predictor_column: str, resource_column: str, run_time_column: str, relative_time_to_failure: float, min_allocation: float):

		self.min_allocation = min_allocation
		self.relative_time_to_failure = relative_time_to_failure

		self.predictor_column = predictor_column
		self.resource_column = resource_column
		self.run_time_column = run_time_column

		# normalize training data

		# we want predictor and target variables in range [0, 1]
		# to process future data and transform data back, we save the linear transformation here
		self.shift = {}
		self.scale = {}
		self.initial_ptp = {}
		self.training_data = training_data.copy()
		for column in [self.predictor_column, self.resource_column]:
			self.shift[column] = np.min(self.training_data[column])
			self.initial_ptp[column] = np.ptp(self.training_data[column])
			self.scale[column] = self.initial_ptp[column] if self.initial_ptp[column] != 0 else 1
		self.unscaled_mean_predictor= training_data[predictor_column].mean()

		self.__transform__(self.training_data)

		# train model
		self.model, self.quality = self.__train__(optimize_base=False)

	def predict(self, data: pd.DataFrame):
		df = data.copy()
		self.__transform__(df)
		df['first_allocation'] = self.model.apply(df)
		self.__inverse_transform__(df)
		return df['first_allocation']

	def __transform__(self, data: pd.DataFrame):
		for column in [self.predictor_column, self.resource_column]:
			data[column] = (data[column] - self.shift[column]) / self.scale[column]

	def __inverse_transform__(self, data: pd.DataFrame):
		for column in [self.predictor_column, self.resource_column]:
			data[column] = data[column] * self.scale[column] + self.shift[column]

	def __predictor_varies_enough__(self):
		return self.initial_ptp[self.predictor_column] > 0.05 * self.unscaled_mean_predictor

	def __quantile_regression__(self, steps: int=5, max_iter=50):
		"""
		Compute slopes and intercepts approximately (low number of iterations) corresponding to different quantile regression lines.
		Uses a quadratic interpolation between 0.5 (median) and 0.9999 to obtain more candidates at the high end of the range.
		"""

		# e.g., [0.99, 0.91, 0.75, 0.51] for steps = 4
		quantile_candidates = [1 - a ** 2 for a in np.linspace(0.01, 0.7, steps)]

		if not self.__predictor_varies_enough__():
			return [self.__linear_model__(slope=0, intercept=self.training_data[self.predictor_column].quantile(q), base=2) for q in quantile_candidates]

		parameters_tried = []

		mod = smf.quantreg('{} ~ {}'.format(self.resource_column, self.predictor_column), self.training_data)

		for quantile in quantile_candidates:

			res = mod.fit(q=quantile, max_iter=max_iter)

			slope_confidence_lower, slope_confidence_upper = res.conf_int().loc[self.predictor_column].tolist()

			intercept = res.params['Intercept']
			slope = slope_confidence_upper

			if np.isnan(slope):
				slope = 0
				intercept = self.training_data[self.resource_column].quantile(quantile)

			parameters_tried.append(self.__linear_model__(slope, intercept, np.nan))

		return parameters_tried

	def __train__(self, optimize_base: bool):

		if not self.__predictor_varies_enough__():
			return self.__train_quantile__()
		else:
			return self.__train_linear__(optimize_base=optimize_base)

	def __train_quantile__(self):
		pass

	def __train_linear__(self, optimize_base: bool = False, max_iter_cobyla=200):
		"""
		Use Constrained Optimization by Linear Approximation to find a good slope, intercept, and optionally, base
		:param data: Needs the following columns 'rss' (peak memory usage), 'run_time', 'input_size' (zero allowed, but not NaN)
		:param initial_solution: A function that returns initial model parameters from a training data set (e.g., initial_solution_zero_max or initial_solution_99percentile)
		:param wastage_func: Computes the over- and under-sizing wastage for a given first allocation (set column 'first_allocation' on the data frame)
		:param optimize_base:
		:param exponential_base: Base for exponential failure handling strategy. If None given, the base is optimized for as well. (E.g., double allocation after each failure, add 50%, etc.)
		:param min_allocation: Minimum memory to allocate to a task. Could be optimized as well.
		:return: the best found model parameters, the according wastage, the wastage function (needed for evaluation set, and changes during optimization if base is not specified), the tried model parameters, and the resulting wastages
		"""

		parameters_tried = []
		wastages_tried = []

		wastage_func = partial(Wastage.exponential, resource_column=self.resource_column, first_allocation_column='first_allocation', run_time_column=self.run_time_column, relative_ttf=self.relative_time_to_failure)

		# compute initial slopes and intercepts
		initial_parameterss = self.__quantile_regression__()

		#
		# add slope from interquartile range
		#

		iqr_predictor = sps.iqr(self.training_data[self.predictor_column])
		print("iqr_predictor: {0}".format(iqr_predictor))
		iqr_resource = sps.iqr(self.training_data[self.resource_column])
		slope = iqr_resource/iqr_predictor if iqr_predictor > 0 else 0
		intercept = self.training_data[self.resource_column].mean() - slope * self.training_data[self.predictor_column].mean()

		initial_parameterss.append(self.__linear_model__(slope, intercept, base=2))

		# constrain base only if it is part of the optimization
		base_constraints = ({'type': 'ineq', 'fun': lambda x: x[2] - self.min_base}) if optimize_base else ()

		def wastage(model_params: [float]):
			params = self.__linear_model__(slope=model_params[0], intercept=model_params[1], base=model_params[2] if optimize_base else 2)

			self.training_data['first_allocation'] = params.apply(self.training_data)

			if optimize_base:
				w = wastage_func(self.training_data, base=model_params[2])
			else:
				w = wastage_func(self.training_data)

			# sometimes the optimizer evaluates infeasible solutions, in which case we do not record the solution
			if not optimize_base or optimize_base and model_params[2] >= self.min_base:
				parameters_tried.append(params)
				wastages_tried.append(w)

			return w.oversizing + w.undersizing

		for initial_parameters in initial_parameterss:

			optimizer_initialization = [initial_parameters.slope, initial_parameters.intercept]
			# start with base 2 if optimizing the base, otherwise specify only slope and intercept
			if optimize_base:
				optimizer_initialization = optimizer_initialization + [2]

			x_res = spo.minimize(fun=wastage, x0=np.array(optimizer_initialization), method="COBYLA",
								 constraints=base_constraints, options=dict(disp=False, maxiter=max_iter_cobyla))

		# if not x_res.success:
		# 	logger.warning("Warning. COBYLA optimizer failed: {}".format(x_res.message))

		best_parameters, lowest_wastage = max(zip(parameters_tried, wastages_tried), key=lambda tuple: tuple[1].maq)

		return best_parameters, lowest_wastage

	def __linear_model__(self, slope, intercept, base):
		return LinearModel(slope=slope, intercept=intercept, base=base, predictor_column=self.predictor_column, min_allocation=self.min_allocation)


class LinearModel:
	def __init__(self, slope: float, intercept: float, predictor_column: Optional[str] = None, base: Optional[float] = None, min_allocation: Optional[float] = None):
		self.slope = slope
		self.intercept = intercept
		self.min_allocation = min_allocation
		self.base = base
		self.predictor_column = predictor_column

	def apply(self, data: pd.DataFrame):
		return np.clip(data[self.predictor_column] * self.slope + self.intercept, a_min=self.min_allocation, a_max=None)

	def __str__(self):
		return "slope {:.2f} intercept {:.2f} base {:.2f} minimum allocation {:.2f}".format(self.slope, self.intercept, self.base, self.min_allocation)


if __name__ == '__main__':
	import time

	df = pd.read_csv("../sample_data/sample.csv.gz")

	REL_TTF = .5
	BASE = 2
	MIN_ALLOC = 0.01

	before = time.time()

	training = df.sample(frac=0.15)
	lwr = LowWastageRegression(training, predictor_column='input_size', resource_column='rss', run_time_column='run_time', relative_time_to_failure=0.5, min_allocation=0.01)

	print("best_params: {0}".format(lwr.model))
	print("maq: {:.2f}% failures: {:.2f}%".format(lwr.quality.maq * 100, lwr.quality.failures / df.size * 100))
	print("\ntime needed: {0}".format(time.time() - before))

	evaluation = df.drop(training.index)
	evaluation['first_allocation'] = lwr.predict(evaluation)
	w = Wastage.exponential(evaluation, 0.5, resource_column='rss', first_allocation_column='first_allocation', run_time_column='run_time')
	print(w)

	training.plot.scatter(x='input_size', y='rss')
	import matplotlib.pyplot as plt
	plt.show()

	evaluation.plot.scatter(x='input_size', y='rss')
	import matplotlib.pyplot as plt
	plt.show()

# train_witt_sampling(df, wastage_func=lambda df: wastage_exponential(df, relative_ttf=0.5, base=2.0), sample=10, plot=True)

