import mkidcalculator as mc
import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.signal import find_peaks, peak_widths

#directory = './data'

# expected energy peaks in eV
e_high = 6490
e_low = 5900
e_peaks = np.array([e_low, e_high])

#loop = mc.Loop.from_pickle(directory + "/analysis/loop_combined.p")
#traces = loop.pulses[0].p_trace

def benchmarkEnergies(traces):
	# calculate pulse energies here
	values = np.sum((traces - np.median(traces, axis=1, keepdims=True)), axis=1)

	return distToEV(values)

def distToEV(values, peaks=e_peaks, drawPlot=False):
	value_space  = np.linspace(np.amin(values), np.amax(values), 1000)
	kernel = stats.gaussian_kde(values)
	values_dist = kernel(value_space)
	peak_indices, properties = find_peaks(values_dist, prominence=0, height=0)
	heights = properties["peak_heights"]

	# conversions from value space to energy space
	e_scale = np.abs((e_peaks[1] - e_peaks[0])/(value_space[peak_indices[0]] - value_space[peak_indices[1]]))
	
	
	if heights[0] > heights[1]:
		e_first = e_peaks[0] - (value_space[peak_indices[0]] * e_scale)
	else:
		e_scale = -e_scale
		e_first = e_peaks[1] - (value_space[peak_indices[0]] * e_scale)

	energies = values*e_scale + e_first

	if drawPlot:
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.hist(energies, bins='auto')
		ax.set_xlabel("Energy [eV]")
		ax.set_ylabel("Counts")
		ax.set_title("Distribution of Energies")
		plt.show()

	return energies

	
def getFWHM(data, drawPlot=False, peaks=e_peaks):
	# find peaks in data
	x = np.linspace(np.amin(data), np.amax(data), 1000)
	kernel = stats.gaussian_kde(data)
	dist = kernel(x)
	peak_indices, properties = find_peaks(dist, prominence=0, height=0)

	# calculate half-max height as percentage relative to prominence for each peak
	prominences = properties["prominences"]
	heights = properties["peak_heights"]
	halfmax_adj = 1-(((0.5*heights)-(heights-prominences))/prominences)

	# conversion from samples to eV
	slope = np.abs((x[peak_indices[0]] - x[peak_indices[1]])/(peak_indices[0] - peak_indices[1]))
	intercept = x[peak_indices[0]] - (peak_indices[0] * slope)

	# create array to fill with fwhm data
	#wshape = np.vstack(peak_widths(energies_dist, peak_indices, rel_height=halfmax_adj[0])).shape
	widths = np.zeros(shape = (4,2))

	# compute fwhm
	widths[:,0] = np.vstack(peak_widths(dist, np.array([peak_indices[0]]), rel_height=halfmax_adj[0]))[:,0]
	widths[:,1] = np.vstack(peak_widths(dist, np.array([peak_indices[1]]), rel_height=halfmax_adj[0]))[:,0]

	# compute variance
	fwhm = slope*widths[0]
	#print("Estimated FWHM of Peak #1: {:.4f} Units".format(fwhm[0]))
	#print("Estimated FWHM of Peak #2: {:.4f} Units".format(fwhm[1]))
	
	if drawPlot:
		# plot
		plt.plot(x, dist)
		plt.plot(peak_indices*slope + intercept, dist[peak_indices], "x")
		plt.hlines(widths[1], widths[2]*slope+intercept, widths[3]*slope+intercept, color="C2")
		plt.show()

	return fwhm

def compareDist(data1, data2, nbin=300, drawPlot='True'):
	fwhm1 = getFWHM(data1)
	fwhm2 = getFWHM(data2)	
	
	print("Data1:")
	print("Estimated FWHM of Peak #1: {:.4f} Units".format(fwhm1[0]))
	print("Estimated FWHM of Peak #2: {:.4f} Units".format(fwhm1[1]))
	print("\nData2:")
	print("Estimated FWHM of Peak #1: {:.4f} Units".format(fwhm2[0]))
	print("Estimated FWHM of Peak #2: {:.4f} Units".format(fwhm2[1]))

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.hist(data1, nbin, alpha=0.5, label="FWHMs: [{:.4f}, {:.4f}]".format(*fwhm1))
	ax.hist(data2, nbin, alpha=0.5, label="FWHMs: [{:.4f}, {:.4f}]".format(*fwhm2))
	ax.legend(loc='upper right')
	
	plt.show()
	

