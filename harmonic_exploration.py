import fractions
from itertools import combinations, accumulate
import operator
import numpy as np
import pandas as pd

def create_harmonic_dataframe(base_freq=256, n_partials=32, verbose=False):
	partials = lambda f, n=16: [f*i for i in range(1, n+1)] 
	get_frac  = lambda fc, ff: fractions.Fraction.from_float(fc / ff).limit_denominator()

	# Find all combinations (no duplicates) of indices of a given iterable `x`
	index_combinations = lambda x: list(
		combinations(range(len(x) if hasattr(x, "__len__") else int(x)), 2)
	)
	octave_above_fundamental   = lambda fc, ff: int(np.log2(fc) - np.log2(ff))
	multiple_above_fundamental = lambda fc, ff: 2 ** (np.log2(fc) - np.log2(ff))

	x = partials(base_freq, n_partials)
	idx = index_combinations(x)

	fracs  = [get_frac(x[j], x[k]) for k,j in idx]
	ifracs = [get_frac(x[k], x[j]) for k,j in idx]

	ratio  = [(x[j] / x[k]) for k,j in idx]
	iratio = [(x[k] / x[j]) for k,j in idx]	
	pairs  = [(x[j], x[k])  for k,j in idx]

	octave = [octave_above_fundamental(x[j], x[k]) + 1 for k,j in idx]
	multip = [multiple_above_fundamental(x[j], x[k]) for k,j in idx]

	df = pd.DataFrame(
		{
			'base_freq': unzip(pairs)[1], 
			'curr_freq': unzip(pairs)[0],		
			'frac': fracs, 
			'ratio': ratio, 
			'inverse_frac': ifracs,
			'inverse_ratio': iratio,
			'multiple': multip,
			'rel_octave': octave,
		}
	)
	logff = np.log2(df.iloc[0].base_freq)
	df['abs_octave'] = df['curr_freq'].apply(lambda x: 1 + int(np.log2(x) - logff))

	if verbose:
		print(df.head())
	return df


def group_harmonic_dataframe(df=None, base_freq=256, n_partials=32):
	accum_add = lambda x: list(accumulate(x, operator.add))
	shift = lambda x, n=1: np.roll(x, n)

	if df is None:
		df = create_harmonic_dataframe(base_freq, n_partials)

	n_partials = int(max(df['ratio']))

	starts = [0] + accum_add(list(range(n_partials - 1, 1, -1)))
	ends = shift(starts, -1)
	ends[-1] = len(df)

	dfs = []
	for start, end in zip(starts, ends):
		sub_df = df.iloc[start:end]
		dfs.append(sub_df)
	return dict(zip(['partial_{}'.format(i) for i in range(n_partials - 1)], dfs))







