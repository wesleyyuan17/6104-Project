import pandas as pd
import numpy as np

TICKERS = ['SPY', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLV']

def main():
	for tic in TICKERS:
		# read in data and do basic data manipulations
		data = pd.read_csv('../data/{}.csv'.format(tic))
		data['Date'] = pd.to_datetime(data['Date'])
		data.set_index('Date', inplace=True)
		data['Returns'] = np.log(data['Adj Close']).diff()

		# split into train and validation sets
		train = data[:'2015-01-01']
		val = data['2015-01-01':]

		train.to_csv('../data/{}_train.csv'.format(tic.lower()))
		val.to_csv('../data/{}_validation.csv'.format(tic.lower()))

if __name__ == '__main__':
	main()