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

	return values

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

	
def getVariances(energies, drawPlot=False, peaks=e_peaks):
	# find peaks in data
	x = np.linspace(np.amin(energies), np.amax(energies), 1000)
	kernel = stats.gaussian_kde(energies)
	energies_dist = kernel(x)
	peak_indices, properties = find_peaks(energies_dist, prominence=0, height=0)

	# calculate half-max height as percentage relative to prominence for each peak
	prominences = properties["prominences"]
	heights = properties["peak_heights"]
	halfmax_adj = 1-(((0.5*heights)-(heights-prominences))/prominences)

	# conversion from samples to eV
	e_scale = (e_peaks[1] - e_peaks[0])/np.abs(peak_indices[0] - peak_indices[1])
	e_first = e_peaks[0] - (peak_indices[0] * e_scale)

	# create array to fill with fwhm data
	#wshape = np.vstack(peak_widths(energies_dist, peak_indices, rel_height=halfmax_adj[0])).shape
	widths = np.zeros(shape = (4,2))

	# compute fwhm
	widths[:,0] = np.vstack(peak_widths(energies_dist, np.array([peak_indices[0]]), rel_height=halfmax_adj[0]))[:,0]
	widths[:,1] = np.vstack(peak_widths(energies_dist, np.array([peak_indices[1]]), rel_height=halfmax_adj[0]))[:,0]

	# compute variance
	var = (e_scale*widths[0])/(2*np.sqrt(2*np.log(2)))
	print("Estimated Variance of Peak #1: {:.4f} eV".format(var[0]))
	print("Estimated Variance of Peak #2: {:.4f} eV".format(var[1]))
	
	if drawPlot:
		# plot in energy space
		e_space = np.linspace(e_first, e_first+e_scale*999, 1000)
		plt.plot(e_space, energies_dist)
		plt.plot(peak_indices*e_scale + e_first, energies_dist[peak_indices], "x")
		plt.hlines(widths[1], widths[2]*e_scale+e_first, widths[3]*e_scale+e_first, color="C2")
		plt.show()
