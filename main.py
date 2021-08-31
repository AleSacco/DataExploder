"""
DATA EXPLODER FOR PRINCIPAL COMPONENT ANALYSIS
v1.0 (09/2021)
For inquiries write to: Alessio Sacco (a.sacco@inrim.it)



This script consists in:
- This main "DataExploder.py" file, to be run;
- (OPTIONAL) a configuration file: "config.py".

The script looks for a "config.py" file, in the directory in which the script is run. If the file is not found,
the script uses default parameters, which are then written in a config file which is created by default if not found.



This script takes as input 2 .csv files describing data as tables:
- a file containing the best estimate for each data point (default file name: data.csv);
- a file containing an uncertainty value for each data point (dafault file name: uncertainties.csv).

A specific data point complete information is contained at a specific table coordinate which is the same in both
files: the estimate file contains the best estimate for that data point, while the uncertainty file contains a value
pertaining the uncertainty. The script also accepts data measured as below the limit of detection (LOD), in which case
the best estimates table entry is to contain the LOD of the measurement (NOT the value 0), while the corresponding
uncertainties entry can contain a blank value, or any non-numerical string to indicate that the first value is a LOD;
the number 0 can also be used for this purpose, but this is not recommended.

Both tables must have the exact same structure in terms of row/column positions, number of label columns, etc.
THE FIRST ROW must be the same for both tables, containing the unique names for each of the columns
(variable names or types of label) and will not be treated as data.
In the configuration file, "Number of label columns" is an integer indicating the number of label columns,
i.e. the number of leftmost columns that will be ignored in the Monte Carlo data generation:
these entries in each row will be replicated verbatim for the corresponding generated samples. These can include the
sample names and/or categorical variables, intended for later analysis.



In this version of the Data Exploder, each single datum consists in two inputs: best estimate
(either a float or non-numeric string, such as "N/A" or "nn", or a 0-length string) and uncertainty. If non-numeric
strings are found the best estimates table, the corresponding variable will be IGNORED FOR ALL DATA.
If a numeric, non-zero uncertainty input is present in the correspondent file, the script interprets it as half of the
confidence interval on the measurement with a Gaussian probability density function (pdf), i.e. expanded uncertainty;
if an uncertainty input is a string, NaN (not a number), or zero, this is interpreted as an indication that the datum
is to be read as BELOW THE LIMIT OF DETECTION: a uniform pdf is used for the data point instead, ranging from zero to
the value indicated in the best estimate table.

Using the appropriate pdf, the script then "explodes" each datum (generates Monte Carlo samples) accordingly, using for
the Gaussian pdfs a coverage factor, usually named "k", which changes according to the choice of confidence level.
As default, in this script k=1.96, corresponding to a confidence level of 95% for a Gaussian pdf, but his can be
changed in the config file.

"""

# TODO: finire di scrivere visualizzazione dati con widget
# TODO: cercare di far scrivere il config file in maniera piu' leggibile invece che tutto su una riga sola

import importlib
import os
import sys
from decimal import Decimal
import numpy as np
import pandas as pd
from scipy.signal import convolve2d
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import dunn_sklearn
import sklearn.metrics as sklm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import matplotlib.widgets as mplwidgets



# Default configuration
config_default = {
	'k': 1.96,
	'Measurements file name': 'data.csv',
	'Uncertainties file name': 'uncertainties.csv',
	'Destination file name': 'Exploded data.csv',
	'N/A mark': 'ND',
	'Number of samples': 1E3,
	'Number of label columns': 3,
	'Skip generating new exploded datafile if "Destination file name" already exists' : True
}



def LoadConfigFile(module):  # Loads config.py if it exists, or it creates one with values of config_default if not
	try:
		confmod = importlib.import_module(module)
	except ModuleNotFoundError:
		config = config_default
		CreateDefaultConfigFile()
		print(TextColors.WARNING + 'Configuration file "config.py" cannot be found. '
			'Default values were loaded and configuration file was created with these values.' + TextColors.ENDC)
	else:
		config = confmod.config
	return config

def CreateDefaultConfigFile():
	with open('config.py', 'w') as f:
		f.write('config = ' + str(config_default))
	return

def LoadConfig(config):  # Closure to make LoadConfigVar() use config data
	def LoadConfigVar(varName):  # Function to load a variable from config file without risks
		try:
			var = config[varName]
		except KeyError:
			print(TextColors.FAIL + 'Configuration file "config.py" appears to be damaged or invalid. '
									'The present "config.py" will be renamed "config_old.py".' + TextColors.ENDC)
			if os.path.isfile('config_old.py'):
				print(TextColors.WARNING + '"config_old.py" already esists. '
						'Renaming it to "config_old_old.py", overwriting if necessary.'
					  + TextColors.ENDC)
				if os.path.isfile('config_old_old.py'):
					os.remove('config_old_old.py')
				os.rename('config_old.py', 'config_old_old.py')
			os.rename('config.py', 'config_old.py')
			print('Restarting script to load default values and to create default "config.py" configuration file...\n')
			os.execv(sys.executable, [sys.executable, '"' + sys.argv[0] + '"'] + sys.argv[1:])
		return var
	return LoadConfigVar

def LoadData(measurementsCsv, uncertaintiesCsv):
	# Read data
	try:
		data = pd.read_csv(measurementsCsv, dtype='str')
		uncertainties = pd.read_csv(uncertaintiesCsv, dtype='str')
	except FileNotFoundError as ex:
		print(TextColors.FAIL + 'Error! ' + str(ex) + TextColors.ENDC)
		sys.exit(-1)
	# Check if data and uncert matrices have the same shape
	if data.shape != uncertainties.shape:
		print(TextColors.FAIL
			  + 'Error! '
			  + measurementsCsv + ' (shaped ' + str(data.shape) + '), and ' + uncertaintiesCsv + ' (shaped ' +
			  str(uncertainties.shape) + ') have different shapes.' + TextColors.ENDC)
		sys.exit(-1)
	return data, uncertainties

def DataClean(data, uncertainties, NAmark):
	# Finds <NAmark> strings in data and drops the corresponding columns in data and uncertainties
	NDmask = data[data == NAmark].dropna(axis=1, how='all')
	dataClean = data.drop(NDmask.columns, axis=1)
	uncertaintiesClean = uncertainties.drop(NDmask.columns, axis=1)
	return dataClean, uncertaintiesClean

def CutToSignificantDigits(x: str, roundTo=2) -> (int, Decimal):  # Works unexpectedly if string is not int or decimal
	"""
	Takes a string containing an integer or a decimal and approximates it to <roundTo> significant digits.
	This is useful to avoid a needlessly large exploded file, which would contain lots of insignificant digits that
	can double or triple the file size without carrying any useful information.
	<roundTo> parameter should not be too small, lest introducing "binning" issues.
	In the script, this function is used to establish from the uncertainty value (if present) the correct number of
	rounding digits for the exploded data. For example, if a datum and its respective uncertainty are given as
	1.2345 and 0.1234 respectively, with roundTo=2 the script generates random data and rounds them as
	[1.23, 1.31, 1.15...] because they are rounded to the decimal corresponding to 2 significant digits of the
	uncertainty (i.e. 0.12).
	Args:
		x (str): The input string containing a number. Works unexpectedly if this string is not int or decimal.
		roundTo (int): The number of significant digits to which to round the datum x.

	Returns:
		decimals (int): the number of decimals to which the datum was rounded. Negative if rounded to tens, hundreds...
		roundNum (Decimal): the rounded number.
	"""
	magicString = '{0:.'+str(roundTo)+'g}'
	roundNum = '{0}'.format(float(magicString.format(Decimal(x))))
	roundNum = roundNum.rstrip('0').rstrip('.')
	split = roundNum.split('.')
	if len(split) == 1:
		decimals = -(len(split[0])-len(split[0].strip('0')))
	else:
		decimals = len(split[1])
	return decimals, roundNum

def ExplodeData(data, uncertainties, sampleNum=1E4, labelColumnsNum=1):
	"""
	Takes "data" and "uncertainties" dataframes and draws <sampleNum> from a distribution whose function and parameters
	depend on the datum and uncertainty:
	- if the uncertainty exists as an integer or decimal, the employed distribution is a Gaussian with mean=<datum> and
	st.dev. = abs(<uncertainty>)/k, where k is the coverage factor indicated in "config.py";
	- if the uncertainty does not exist (i.e. is not numeric) or is zero for that datum, it defaults to a uniform
	distribution ranging from 0 to <datum>*2. The omission of the uncertainty value is to indicate that the datapoint
	is below the limit of detection (LOD): in this case, <datum> indicates LOD/2 for that point, i.e. the best estimate.

	The function cycles through rows and columns of data matrix and creates Monte Carlo samples for each as described,
	then creates the entire DataFrame.
	Args:
		data (dataframe): DataFrame containing data (best estimates for each measurement).
		uncertainties (dataframe): DataFrame containing uncertainties if available,
		or either 0 or a string different from "NAmark"	to indicate that the corresponding datapoint was measured
		below the limit of detection, triggering a uniform distribution for sample generation for the datum in question.
		sampleNum (int, float): Number of samples to generate for each datapoint. If float, it gets converted to int.
		labelColumnsNum (int): Number of non-data columns starting from the left (e.g. name columns, labels...).

	Returns:
		explodedData: DataFrame containing <sampleNum> draws for each measurement and their corresponding label columns.
	"""
	sampleNum = int(sampleNum)
	frames = []
	columnsNames = data.columns
	for index, row in tqdm(data.iterrows(), total=data.shape[0], unit=' samples'):
		df = pd.DataFrame(index=range(sampleNum), columns=columnsNames)
		for col in columnsNames[:labelColumnsNum]:  # Write label columns
			df[col] = row[col]
		for col in columnsNames[labelColumnsNum:]:  # Generate random numbers
			if not isinstance(pd.to_numeric(uncertainties[col][index], errors='ignore'), (int, float)) \
					or pd.isnull(uncertainties[col][index]) or uncertainties[col][index] == 0:
				# Uniform distribution from 0 to 2*value if uncertainty is string or NaN or zero
				rounding = CutToSignificantDigits(data[col][index])[0]
				rands = np.around(np.random.uniform(0, 2 * pd.to_numeric(data[col][index]), sampleNum),
								  decimals=rounding)
			else:
				# Gaussian distribution if uncertainty is a number different than 0
				rounding = CutToSignificantDigits(uncertainties[col][index])[0]
				rands = np.around(
					np.random.normal(pd.to_numeric(data[col][index]), pd.to_numeric(uncertainties[col][index]) / 1.96,
									 sampleNum), decimals=rounding)
			df[col] = rands
		frames.append(df)
	print('Data generation complete. Exporting data...')
	explodedData = pd.concat(frames, ignore_index=True)
	return explodedData

def SaveExplodedData(data, filename):
	data.to_csv(filename, index=False)
	print(TextColors.GREEN +
		  'Data successfully exported to file: "' + filename + '".' +
		  TextColors.ENDC)
	return

def LoadExplodedData(filename):
	loadedData = pd.read_csv(filename)
	print('Exploded data loaded from existing file.')
	return loadedData

def PCAonExplodedDataFromCsv(explodedDataFileName):  #TODO: scrivere nel docstring che si puo' chiamare questa da prompt
	data = pd.read_csv(explodedDataFileName)
	PCAonExplodedData(data)
	return

def PCAonOriginalData():
	# Load data
	config = LoadConfigFile('config')
	LoadConfigVariable = LoadConfig(config)
	measurementsFileName = LoadConfigVariable('Measurements file name')
	uncertaintiesFileName = LoadConfigVariable('Uncertainties file name')
	NAmark = LoadConfigVariable('N/A mark')
	numberOfLabelColumns = LoadConfigVariable('Number of label columns')
	data_DF, uncertainties_DF = LoadData(measurementsFileName, uncertaintiesFileName)
	# Do PCA
	data_DF, uncertainties_DF = DataClean(data_DF, uncertainties_DF, NAmark=NAmark)
	data = data_DF.iloc[:, numberOfLabelColumns:]
	dataLabels = data_DF.iloc[:, :numberOfLabelColumns]
	scaler = StandardScaler(with_mean=True, with_std=True)
	pca = PCA(n_components=3)
	dataPP = scaler.fit_transform(data)
	dataPCA = pca.fit_transform(dataPP)
	dataPCA = pd.concat([dataLabels, pd.DataFrame(dataPCA)], axis=1)
	loadings = pd.DataFrame(pca.components_, columns=data.columns)
	print('Explained variances (%): ' + str(pca.explained_variance_ratio_*100) +
		  ' (total: ' + str(round(sum(pca.explained_variance_ratio_*100), 2)) + '%).')
	# Visualize data
	plt.figure()
	xAxisPC = 2  # PC to be visualized on x axis, 1-indexed
	yAxisPC = 3  # PC to be visualized on y axis, 1-indexed
	labelsCol = 2  # Label column as colors for markers, 1-indexed
	explodedDataPCA_grouped = dataPCA.groupby(dataPCA.columns[labelsCol])
	for key, group in explodedDataPCA_grouped:
		plt.scatter(group.iloc[:, xAxisPC-1+numberOfLabelColumns], group.iloc[:, yAxisPC-1+numberOfLabelColumns],
					label=key)
	plt.xlabel('PC' + str(xAxisPC) + ' (' + str(round(pca.explained_variance_ratio_[xAxisPC-1]*100, 2)) + '%)')
	plt.ylabel('PC' + str(yAxisPC) + ' (' + str(round(pca.explained_variance_ratio_[yAxisPC-1]*100, 2)) + '%)')
	plt.title('Original data PCA')
	plt.legend()
	return loadings, dataPCA

class TextColors:
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'
	FAIL = '\033[91m'
	GREEN = '\033[92m'
	WARNING = '\033[93m'
	BLUE = '\033[94m'
	HEADER = '\033[95m'
	CYAN = '\033[96m'



### Scripts


def DataExploder():

	## Load configuration variables and data
	measurementsFileName = LoadConfigVariable('Measurements file name')
	uncertaintiesFileName = LoadConfigVariable('Uncertainties file name')
	destinationFileName = LoadConfigVariable('Destination file name')
	NAmark = LoadConfigVariable('N/A mark')
	numberOfSamples = LoadConfigVariable('Number of samples')
	numberOfLabelColumns = LoadConfigVariable('Number of label columns')
	skipGenerateNewExplodedFile = LoadConfigVariable(
			'Skip generating new exploded datafile if "Destination file name" already exists')

	# Clean data, generate samples and save them to file, if this file does not exist or this ckeck is ignored by config
	if (not skipGenerateNewExplodedFile) or (not os.path.isfile(destinationFileName)):
		data, uncertainties = LoadData(measurementsFileName, uncertaintiesFileName)
		data, uncertainties = DataClean(data, uncertainties, NAmark=NAmark)
		explodedData = ExplodeData(data, uncertainties, sampleNum=numberOfSamples, labelColumnsNum=numberOfLabelColumns)
		SaveExplodedData(explodedData, destinationFileName)
	else:
		print(destinationFileName + ' already exists. Skipping generation of data. This behavior can be modified '
									'in the configuration file.')
		explodedData = LoadExplodedData(destinationFileName)

	return explodedData


def PCAonExplodedData(explodedDataframe, labelColumnsNum=None):
	if labelColumnsNum == None: labelColumnsNum = LoadConfigVariable('Number of label columns')

	# Do PCA
	explodedData = explodedDataframe.iloc[:, labelColumnsNum:]
	explodedDataLabels = explodedDataframe.iloc[:, :labelColumnsNum]
	scaler = StandardScaler(with_mean=True, with_std=True)
	pca = PCA(n_components=3)
	print('Scaling data...')
	explodedDataPP = scaler.fit_transform(explodedData)
	print('Scaling complete.')
	print('Performing PCA on data (number of components = ' + str(pca.n_components) + ')...')
	explodedDataPCA = pca.fit_transform(explodedDataPP)
	print('PCA complete.')

	# Recompose DataFrame
	explodedDataPCA = pd.concat([explodedDataLabels, pd.DataFrame(explodedDataPCA)], axis=1)
	loadings = pd.DataFrame(pca.components_, columns=explodedData.columns)

	# Visualize data
	PlotExplodedPCA(pca, explodedDataPCA, loadings)

	return loadings, explodedDataPCA



def PlotExplodedPCA(pca, explodedDataPCA, loadings, numberOfLabelColumns=None):

	if numberOfLabelColumns == None: numberOfLabelColumns = LoadConfigVariable('Number of label columns')
	print('Explained variances (%): ' + str(pca.explained_variance_ratio_ * 100) +
		  ' (total: ' + str(round(sum(pca.explained_variance_ratio_ * 100), 2)) + '%).')

	## Graphs parameters ########################################################################################
	xAxisPC = 2  # PC to be visualized on x axis in 2D plots, 1-indexed
	yAxisPC = 3  # PC to be visualized on y axis in 2D plots, 1-indexed
	colorLabel = 2  # Label column as colors
	slicingStep = 10  # In scatter plots, graph one point every <slicingStep>
	# colorsCycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
	colorsCycle = ['#000000', '#FFAAAA', '#FF0000', '#AAAAAA']

	## Figure 1: loadings
	figure1 = plt.figure()
	for i in range(loadings.shape[0]):
		plt.plot(loadings.iloc[i], label='PC' + str(i + 1))
		plt.legend()

	return

	## Figure 2: 2D score plot (plot one point every <slicingStep>)
	# figure2 = plt.figure()

	def Refresh2Dscatter():
		ax2.clear()
		ax2.set_title('Exploded data PCA')
		for num, (key, group) in enumerate(explodedDataPCA_grouped):
			ax2.scatter(group.iloc[0::slicingStep, xAxisPC - 1 + numberOfLabelColumns],
							group.iloc[0::slicingStep, yAxisPC - 1 + numberOfLabelColumns],
							label=key, alpha=.05, color=colorsCycle[num])
		ax2.set_xlabel(
			'PC' + str(xAxisPC) + ' (' + str(round(pca.explained_variance_ratio_[xAxisPC - 1] * 100, 2)) + '%)')
		ax2.set_ylabel(
			'PC' + str(yAxisPC) + ' (' + str(round(pca.explained_variance_ratio_[yAxisPC - 1] * 100, 2)) + '%)')
		ax2.legend()
		figure2.canvas.draw_idle()
		return

	def changePCx_2Dscatter(label):
		changePC_2Dscatter(label, axis='x')
		return

	def changePCy_2Dscatter(label):
		changePC_2Dscatter(label, axis='y')
		return

	def changePC_2Dscatter(label, axis='x'):
		nonlocal xAxisPC, yAxisPC
		if axis == 'y':
			yAxisPC = int(label)
		else:
			xAxisPC = int(label)
		Refresh2Dscatter()
		return

	figure2, ax2 = plt.subplots()
	plt.subplots_adjust(left=0.35)
	box2_1 = plt.axes([0.02, 0.5, 0.2, 0.3])
	buttons2_1 = mplwidgets.RadioButtons(box2_1, (1, 2, 3), active=1)
	buttons2_1.on_clicked(changePCx_2Dscatter)

	explodedDataPCA_grouped = explodedDataPCA.groupby(explodedDataPCA.columns[colorLabel])
	Refresh2Dscatter()

	# One of the following two lines is needed in order to make plots visible
	plt.show()
	# plt.show(block=True)


	## Figure 3: 3D score plot (plot one point every <slicingStep>)
	ax = Axes3D(plt.figure())
	explodedDataPCA_grouped = explodedDataPCA.groupby(explodedDataPCA.columns[colorLabel])
	for key, group in explodedDataPCA_grouped:
		ax.scatter(group.iloc[0::slicingStep, 0 + numberOfLabelColumns],
				   group.iloc[0::slicingStep, 1 + numberOfLabelColumns],
				   group.iloc[0::slicingStep, 2 + numberOfLabelColumns],
				   label=key, alpha=.2)
	plt.legend()
	ax.set_xlabel('PC1 (' + str(round(pca.explained_variance_ratio_[0] * 100, 2)) + '%)')
	ax.set_ylabel('PC2 (' + str(round(pca.explained_variance_ratio_[1] * 100, 2)) + '%)')
	ax.set_zlabel('PC3 (' + str(round(pca.explained_variance_ratio_[2] * 100, 2)) + '%)')
	plt.title('Exploded data PCA')

	## Figure 4: 2D score plot (density)
	# ax = Axes3D(plt.figure())
	# # colorsCycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
	# colorsCycle = ['#000000', '#FFAAAA', '#FF0000', '#AAAAAA']
	# patches = []
	# explodedDataPCA_selectCols = pd.concat([explodedDataPCA.iloc[:, [i for i in range(numberOfLabelColumns)]],
	# 										explodedDataPCA.iloc[:, xAxisPC - 1 + numberOfLabelColumns],
	# 									    explodedDataPCA.iloc[:, yAxisPC - 1 + numberOfLabelColumns]],
	# 									   axis=1)
	# binsSpace = np.linspace(explodedDataPCA_selectCols.iloc[:, -2:].min().min(),
	# 				   explodedDataPCA_selectCols.iloc[:, -2:].max().max(),
	# 				   100)  # Number of bins for each axis here ###################################################
	# explodedDataPCA_selectCols_grouped = \
	# 	explodedDataPCA_selectCols.groupby(explodedDataPCA_selectCols.columns[colorLabel])
	# for num, (key, group) in enumerate(explodedDataPCA_selectCols_grouped):
	# 	histogrid = np.histogramdd(group.iloc[:, numberOfLabelColumns:].values,
	# 							  bins=[binsSpace, binsSpace],
	# 							  )  # List with len=2: [0] is the image; [1] is the list of bin limits
	# 	histogrid_vals = histogrid[0]
	# 	# Convert from n+1 edges of bins to n centers of bins
	# 	histogrid_X = np.array([(histogrid[1][0][i+1] + histogrid[1][0][i]) / 2 for i in range(len(histogrid[1][0])-1)])
	# 	histogrid_Y = np.array([(histogrid[1][1][i+1] + histogrid[1][1][i]) / 2 for i in range(len(histogrid[1][1])-1)])
	# 	histogrid_XX, histogrid_YY = np.meshgrid(histogrid_X, histogrid_Y, indexing='ij')  # Indexing is IMPORTANT!
	# 	histogrid_vals_conv = convolve2d(histogrid_vals,  # To color all nonzero facets instead of having some alpha'd
	# 									 np.array([[1,1,0],[1,1,0],[0,0,0]]),
	# 									 mode='same')  # Since "facecolors=" uses indices of facets and ignores the rest
	# 	ax.plot_surface(histogrid_XX, histogrid_YY, histogrid_vals,
	# 						   rstride=1, cstride=1, edgecolor='none',
	# 						   facecolors=np.where(histogrid_vals_conv < 1, '#FFFFFF00', colorsCycle[num]+'AA'))
	# 	# To make faces a single color, use "color=" instead of "facecolors="
	# 	patches.append(mpatches.Patch(color=colorsCycle[num], label=key))
	# ax.set_xlabel('PC' + str(xAxisPC) + ' (' + str(round(pca.explained_variance_ratio_[xAxisPC - 1] * 100, 2)) + '%)')
	# ax.set_ylabel('PC' + str(yAxisPC) + ' (' + str(round(pca.explained_variance_ratio_[yAxisPC - 1] * 100, 2)) + '%)')
	# ax.set_zlabel('Density (instances/bin)')
	# plt.legend(handles=patches)
	# plt.show()

	ax = Axes3D(plt.figure())
	patches = []
	explodedDataPCA_selectCols = pd.concat([explodedDataPCA.iloc[:, [i for i in range(numberOfLabelColumns)]],
											explodedDataPCA.iloc[:, xAxisPC - 1 + numberOfLabelColumns],
											explodedDataPCA.iloc[:, yAxisPC - 1 + numberOfLabelColumns]],
										   axis=1)
	binsSpace = np.linspace(explodedDataPCA_selectCols.iloc[:, -2:].min().min(),
							explodedDataPCA_selectCols.iloc[:, -2:].max().max(),
							100)  # Number of bins for each axis here ###################################################
	explodedDataPCA_selectCols_grouped = \
		explodedDataPCA_selectCols.groupby(explodedDataPCA_selectCols.columns[colorLabel])
	for num, (key, group) in enumerate(explodedDataPCA_selectCols_grouped):
		histogrid = np.histogramdd(group.iloc[:, numberOfLabelColumns:].values,
								   bins=[binsSpace, binsSpace],
								   )  # List with len=2: [0] is the image; [1] is the list of bin limits
		histogrid_vals = histogrid[0]
		# Convert from n+1 edges of bins to n centers of bins
		histogrid_X = np.array(
				[(histogrid[1][0][i + 1] + histogrid[1][0][i]) / 2 for i in range(len(histogrid[1][0]) - 1)])
		histogrid_Y = np.array(
				[(histogrid[1][1][i + 1] + histogrid[1][1][i]) / 2 for i in range(len(histogrid[1][1]) - 1)])
		histogrid_XX, histogrid_YY = np.meshgrid(histogrid_X, histogrid_Y, indexing='ij')  # Indexing is IMPORTANT!
		histogrid_vals_conv = convolve2d(histogrid_vals,  # To color all nonzero facets instead of having some alpha'd
										 np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]]),
										 mode='same')  # Since "facecolors=" uses indices of facets and ignores the rest
		histogrid_vals = np.where(histogrid_vals > 10000, 10000, histogrid_vals)
		ax.plot_surface(histogrid_XX, histogrid_YY, histogrid_vals,
						rstride=1, cstride=1, edgecolor='none',
						facecolors=np.where(histogrid_vals_conv < 1, '#FFFFFF00', colorsCycle[num] + 'AA'))
		# To make faces a single color, use "color=" instead of "facecolors="
		patches.append(mpatches.Patch(color=colorsCycle[num], label=key))
	ax.set_xlabel('PC' + str(xAxisPC) + ' (' + str(round(pca.explained_variance_ratio_[xAxisPC - 1] * 100, 2)) + '%)')
	ax.set_ylabel('PC' + str(yAxisPC) + ' (' + str(round(pca.explained_variance_ratio_[yAxisPC - 1] * 100, 2)) + '%)')
	ax.set_zlabel('Density (instances/bin)')
	plt.legend(handles=patches)
	plt.show()

	return


def CalculateMetrics(dataframe, metric):
	method = getattr(sklm, metric)
	labelsCol = 2
	data = dataframe.iloc[:, numberOfLabelColumns:]
	labels = dataframe.iloc[:, labelsCol]
	score = method(data, labels)
	print(metric + ': ' + str(score) + '.')
	return score


def Dunn(dataframe):
	labelsCol = 2
	data = np.array(dataframe.iloc[:, numberOfLabelColumns:])
	labels = np.array(dataframe.iloc[:, labelsCol])
	distances = sklm.pairwise.euclidean_distances(data)
	score = dunn_sklearn.dunn(labels, distances)
	print('dunn_index: ' + str(score) + '.')
	return score





if __name__ == '__main__':

	# Load configuration file
	config = LoadConfigFile('config')
	LoadConfigVariable = LoadConfig(config)

	# Function to do PCA on origial data if needed
	loadingsOriginal, originalDataPCA = PCAonOriginalData()

	# Data Exploder
	explodedData = DataExploder()

	# PCA and plots on exploded data
	loadings, explodedDataPCA = PCAonExplodedData(explodedData)

	# Cluster metrics
	numberOfLabelColumns = LoadConfigVariable('Number of label columns')
	# metrics = ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score']
	# print('\nOriginal data metrics')
	# for m in metrics:
	# 	CalculateMetrics(originalDataPCA, m)
	# Dunn(originalDataPCA)
	# print('\nExploded data metrics')
	# for m in metrics:
	# 	CalculateMetrics(explodedDataPCA, m)
	# Dunn(explodedDataPCA)
