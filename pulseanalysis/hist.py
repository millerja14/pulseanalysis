import mkidcalculator as mc
import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.signal import find_peaks, peak_widths
from scipy import interpolate

import pulseanalysis.data as mkid

A=8.2
B=16.2

loc1 = 5887.65
loc2 = 5898.75
loc3 = 6490.45
loc0 = (A*loc1+B*loc2)/(A+B)

e_low = loc0
e_high = loc3
e_cutoff = 6250
e_peaks = np.array([e_low, e_high])




#loop = mc.Loop.from_pickle(directory + "/analysis/loop_combined.p")
#traces = loop.pulses[0].p_trace

def benchmarkEnergies(traces=None):
	
	if not isinstance(traces, (np.ndarray)):
		print("benchmarkEnergies(): No traces given, getting default traces...")
		traces = mkid.loadTraces()
	
	# calculate pulse energies here
	values = np.sum((traces - np.median(traces, axis=1, keepdims=True)), axis=1)

	return distToEV(values)

def getDoublePeak_fe55(data, drawPlot=False):
	cutoff = getCutoffs(data, 2)
	print(cutoff)

	doublepeak = data[data<cutoff]

	if drawPlot:
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.hist(doublepeak, bins='auto')
		ax.set_xlabel("Energy [eV]")
		ax.set_ylabel("Counts")
		ax.set_title("Fe55 5.9 keV Double Peak")
		plt.show()
	
	return doublepeak


def resolveDoublePeak(data=None, x0=loc1, x1=loc2, A=A, B=B, loops=2, bw=None, drawPlot=False):
	
	if data is None:
		data = getDoublePeak_fe55(benchmarkEnergies())
		cutoff = getCutoffs(data, 2)
		data = data[data<cutoff]

	kernel = stats.gaussian_kde(data)
	if bw is not None:
		kernel.set_bandwidth(bw_method=bw)
	bw = kernel.factor	

	total = A+B
	A = A/total
	B = B/total

	dx = x1-x0

	g_array_full = np.array([])
	x_g_full = np.array([])

	f_array_full = np.array([])
	x_f_full = np.array([])

	for i in range(loops):
		start = np.amin(data)
		stop = np.amax(data)
		step = abs(dx)	
		start = start + i*(step/loops)

		x_g = np.arange(start, stop, step)
		g_array = kernel(x_g)
		N = g_array.size - 1	
	
		x_f_1 = np.arange(start-x1, stop-x1, 2*step)
		x_f_2 = np.arange(start-x0, stop-x0, 2*step)
		x_f = np.empty((x_f_1.size + x_f_2.size), dtype=x_f_1.dtype)
		x_f[0::2] = x_f_1
		x_f[1::2] = x_f_2


		M = x_f.size
		if M < N+2:
			x_g = x_g[:M-1]
			g_array = g_array[:M-1]
			N = M-2
		elif M > N+2:
			x_f = x_f[:N+2]
	
	
		matrix = np.zeros((N+1, N+2))
		for i in range(N+1):
			matrix[i,i] = B
			matrix[i,i+1] = A

		f_array = np.linalg.lstsq(matrix, g_array)[0]

		#print("g_array size: ", g_array.size)
		#print("f_array size: ", f_array.size)
		#print("matrix shape: ", matrix.shape)

		g_array_full = np.append(g_array_full, g_array)
		x_g_full = np.append(x_g_full, x_g)

		f_array_full = np.append(f_array_full, f_array)
		x_f_full = np.append(x_f_full, x_f)

	f_order = np.argsort(x_f_full)
	f_array_full = np.take(f_array_full, f_order)
	x_f_full = np.take(x_f_full, f_order)

	g_order = np.argsort(x_g_full)
	g_array_full = np.take(g_array_full, g_order)
	x_g_full = np.take(x_g_full, g_order)

	if drawPlot==True:
		fig = plt.figure()
		ax = fig.add_subplot(111)
		#ax.hist(data, bins='auto', density=True)
		ax.plot(x_f_full+x0, A*f_array_full, color='blue')
		ax.plot(x_f_full+x1, B*f_array_full, color='blue')
		ax.plot(x_g_full, g_array_full, color='green')
		plt.show()

	g_fwhm = fwhmFromPeak(x_g_full, g_array_full)
	f_fwhm = fwhmFromPeak(x_f_full, f_array_full)

	print("g_fwhm: ", g_fwhm)
	print("f_fwhm: ", f_fwhm)

	return x_f_full, f_array_full

def resolveSinglePeak(data=None, bw=None, samples=1000):
	if data is None:
		data = getDoublePeak_fe55(benchmarkEnergies())
		cutoff = getCutoffs(data, 2)
		data = data[data>=cutoff]

	kernel = stats.gaussian_kde(data)
	if bw is not None:
		kernel.set_bandwidth(bw_method=bw)
	bw = kernel.factor

	xvalues = np.linspace(np.amin(data), np.amax(data), samples)
	yvalues = kernel(xvalues)

	return xvalues, yvalues

def fwhmFromPeak(xvalues, yvalues):
	peak_indices, properties = find_peaks(yvalues, prominence=0, height=0)

	# calculate half-max height as percentage relative to prominence for each peak
	prominences = properties["prominences"]
	heights = properties["peak_heights"]

	# select proper number of largest peaks
	idx = np.argsort(heights)[-1]
	peak_idx = np.take(peak_indices, [idx])[0]
	prominence = np.take(prominences, [idx])[0]
	height = np.take(heights, [idx])[0]
	
	halfmax_adj = 1-(((0.5*height)-(height-prominence))/prominence)

	width_data = peak_widths(yvalues, np.array([peak_indices[idx]]), rel_height=halfmax_adj)
	width_indices = np.rint(np.concatenate((width_data[2], width_data[3]))).astype(int)
	bounds = np.take(xvalues, width_indices)
	fwhm = abs(bounds[1]-bounds[0])
	
	return fwhm, width_data[1], bounds[0], bounds[1]

def distToEV(values, peaks=e_peaks, drawPlot=False):
	cutoff = getCutoffs(values, 2)
	print("peaks: ", peaks)
	print("cutoff: ", cutoff)
	
	values0 = values[values<cutoff]
	values1 = values[values>=cutoff]

	median0 = np.median(values0)
	median1 = np.median(values1)

	e_scale = np.abs((e_peaks[1] - e_peaks[0])/(median0-median1))

	if values0.size > values1.size:
		e_first = e_peaks[0] - (median0 * e_scale)
	else:
		e_scale = -e_scale
		e_first = e_peaks[1] - (median0 * e_scale)
	
	energies = values*e_scale + e_first
	
	return energies

def distToEV_kde(values, peaks=e_peaks, drawPlot=False):

	value_space  = np.linspace(np.amin(values), np.amax(values), 10000)
	kernel = stats.gaussian_kde(values)
	values_dist = kernel(value_space)
	peak_indices, properties = find_peaks(values_dist, prominence=0, height=0)
	heights = properties["peak_heights"]

	# select proper number of largest peaks
	idx = np.sort(np.argsort(heights)[-2:])
	peak_indices = np.take(peak_indices, idx)
	heights = np.take(heights, idx)

	#fig = plt.figure()
	#ax = fig.add_subplot(111)
	#ax.hist(values, bins='auto', density=True)
	#ax.plot(value_space, values_dist)
	#ax.set_title("distToEV KDE fit")
	#plt.show()
	#print("distToEV peaks locations: ", np.take(value_space, peak_indices))
	#print("distToEV peak heights: ", heights)

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

def distToEV_split(values, peaks=e_peaks, bw_list=[.15,.2], samples=10000):
	
	cutoff = getCutoffs(values, 2)
	values0 = values[values<cutoff]
	values1 = values[values>=cutoff]

	xvalues0, yvalues0 = resolveSinglePeak(values0, bw=bw_list[0], samples=samples)
	xvalues1, yvalues1 = resolveSinglePeak(values1, bw=bw_list[1], samples=samples)

	idx0 = np.argsort(yvalues0)[-1]
	idx1 = np.argsort(yvalues1)[-1]

	max0 = np.take(xvalues0, idx0)
	max1 = np.take(xvalues1, idx1)

	height0 = np.take(yvalues0, idx0)
	height1 = np.take(yvalues1, idx1)

	# conversions from value space to energy space
	e_scale = np.abs((e_peaks[1] - e_peaks[0])/(max1 - max0))
	
	
	if height0 > height1:
		e_first = e_peaks[0] - (max1 * e_scale)
	else:
		e_scale = -e_scale
		e_first = e_peaks[1] - (max1 * e_scale)

	energies = values*e_scale + e_first

	return energies

def testBW(data, desc=""):
	
	scale = 1

	x = np.linspace(np.amin(data), np.amax(data), 1000)
	kernel = stats.gaussian_kde(data)
	bw_scott = kernel.factor
	print("Scott BW: {} Peaks: {}".format(kernel.factor, find_peaks(kernel(x))[0].size))	

	dist1 = scale*kernel(x)
	
	kernel.set_bandwidth(bw_method='silverman')
	bw_silverman = kernel.factor	
	print("Silverman BW: {} Peaks: {}".format(kernel.factor, find_peaks(kernel(x))[0].size))	
	dist2 = scale*kernel(x)

	kernel.set_bandwidth(bw_method=.05)
	print("Const BW: {} Peaks: {}".format(kernel.factor, find_peaks(kernel(x))[0].size))	
	dist3 = scale*kernel(x)

	kernel.set_bandwidth(bw_method=.1)
	print("Const BW: {} Peaks: {}".format(kernel.factor, find_peaks(kernel(x))[0].size))
	dist4 = scale*kernel(x)

	kernel.set_bandwidth(bw_method=.15)
	print("Const BW: {} Peaks: {}".format(kernel.factor, find_peaks(kernel(x))[0].size))
	dist5 = scale*kernel(x)

	kernel.set_bandwidth(bw_method=.3)
	print("Const BW: {} Peaks: {}".format(kernel.factor, find_peaks(kernel(x))[0].size))
	dist6 = scale*kernel(x)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.hist(data, bins='auto', density=True)
	ax.plot(x, dist1, label='Scott (default) {:.4f}'.format(bw_scott))
	ax.plot(x, dist2, label='Silverman {:.4f}'.format(bw_silverman))
	ax.plot(x, dist3, label='Const .05', linestyle='dashed', linewidth=2)
	ax.plot(x, dist4, label='Const .10', linestyle='dashed', linewidth=2)
	ax.plot(x, dist5, label='Const .15', linestyle='dashed', linewidth=2)
	ax.plot(x, dist6, label='Const .30', linestyle='dashed', linewidth=2)
	ax.legend()
	
	title = "BW Comparison"
	if not desc == "":
		title = title + ": " + desc
	
	ax.set_title(title)
	
	plt.show()
	
def getFWHM(data, npeaks=None, bw=None, samples=1000, desc="", xlabel="", drawPlot=False):
	'''
	Takes in distribution data, number of samples used to resolve the fit, and
	drawPlot. Outputs array containing fwhm for each peak and the kde distribution
	containing "samples" number of data points.
	'''

	# find peaks in data
	x = np.linspace(np.amin(data), np.amax(data), samples)
	kernel = stats.gaussian_kde(data)

	if bw is not None:
		kernel.set_bandwidth(bw_method=bw)

	bw = kernel.factor

	dist = kernel(x)
	peak_indices, properties = find_peaks(dist, prominence=0, height=0)
	
	if npeaks is None:
		npeaks = peak_indices.size
	elif not isinstance(npeaks, int):
		raise ValueError("Number of peaks must be an integer value.")
	elif not npeaks > 0:
		raise ValueError("Number of peaks must be greater than 0.")
	elif (npeaks > peak_indices.size):
		print("Can only find {0} peaks in the data, Assuming {0} peaks instead of {1}.".format(peak_indices.size, npeaks))
		npeaks = peak_indices.size	
	
	# calculate half-max height as percentage relative to prominence for each peak
	prominences = properties["prominences"]
	heights = properties["peak_heights"]

	# select proper number of largest peaks
	idx = np.sort(np.argsort(heights)[-npeaks:])
	peak_indices = np.take(peak_indices, idx)
	prominences = np.take(prominences, idx)
	heights = np.take(heights, idx)

	halfmax_adj = 1-(((0.5*heights)-(heights-prominences))/prominences)

	# conversion from samples to eV
	slope = np.abs(x[1] - x[0])

	# create array to fill with fwhm data
	#wshape = np.vstack(peak_widths(energies_dist, peak_indices, rel_height=halfmax_adj[0])).shape
	widths = np.zeros(shape = (4, npeaks))

	# compute fwhm
	for i in range(npeaks):
		widths[:,i] = np.vstack(peak_widths(dist, np.array([peak_indices[i]]), rel_height=halfmax_adj[i]))[:,0]
	
	# compute variance
	fwhm = slope*widths[0]
	
	if drawPlot:
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.hist(data, bins='auto', density=True)
		ax.plot(x, dist, label="BW: {:.4f}".format(bw))
		ax.plot(x[peak_indices], dist[peak_indices], "x")
		ax.hlines(widths[1], widths[2]*slope+x[0], widths[3]*slope+x[0], color="C2")
		title = "KDE"
		if not desc == "":
			title = title + ": " + desc
		if not xlabel == "":
			ax.set_xlabel(xlabel)
		plt.show()

	return fwhm, dist

def getCutoffs(data, npeaks, samples=1000):

	# find minima
	x = np.linspace(np.amin(data), np.amax(data), samples)
	kernel = stats.gaussian_kde(data)
	dist = kernel(x)
	cutoff_indices, cutoff_properties = find_peaks(-dist, prominence=0)
	
	# get most prominent minima
	k = npeaks-1
	cutoff_prominences = cutoff_properties["prominences"]
	idx = np.argsort(cutoff_prominences)[-k:]
	cutoff_indices_filtered = np.sort(np.take(cutoff_indices, idx))

	# points at which we will split the data
	cutoffs = x[cutoff_indices_filtered]
	cutoffs = np.sort(cutoffs)
	
	return cutoffs

def getFWHM_fe55(data=None, x0=loc1, x1=loc2, A=A, B=B, loops=20, bw_list=[.15,.2], samples=10000, drawPlot=True):

	total = A+B
	A = A/total
	B = B/total
	
	if data is None:
		data = benchmarkEnergies()

	cutoff = getCutoffs(data, 2)
	data0 = data[data<cutoff]
	data1 = data[data>=cutoff]

	xdata0, ydata0 = resolveSinglePeak(data0, bw=bw_list[0], samples=samples)
	width0, width_height0, left_ips0, right_ips0 = fwhmFromPeak(xdata0, ydata0)

	xdata0x, ydata0x = resolveDoublePeak(data0, x0=x0, x1=x1, A=A, B=B, bw=bw_list[0], loops=loops, drawPlot=drawPlot)
	
	xdata00 = xdata0x + x0
	xdata01 = xdata0x + x1
	ydata00 = A*ydata0x
	ydata01 = B*ydata0x

	width00, width_height00, left_ips00, right_ips00 = fwhmFromPeak(xdata00, ydata00)
	width01, width_height01, left_ips01, right_ips01 = fwhmFromPeak(xdata01, ydata01)

	xdata10, ydata10 = resolveSinglePeak(data1, bw=bw_list[1], samples=samples)
	width10, width_height10, left_ips10, right_ips10 = fwhmFromPeak(xdata10, ydata10)

	#y_peak00 = np.amax(ydata00)
	#x_peak00 = np.take(xdata00, np.argsort(ydata00)[-1])

	#y_peak01 = np.amax(ydata01)
	#x_peak01 = np.take(xdata01, np.argsort(ydata01)[-1])
	
	#y_peak10 = np.amax(ydata10)
	#x_peak10 = np.take(xdata10, np.argsort(ydata10)[-1])i
	
	f0 = interpolate.interp1d(xdata0, ydata0)
	f00 = interpolate.interp1d(xdata00, ydata00)
	f01 = interpolate.interp1d(xdata01, ydata01)
	f10 = interpolate.interp1d(xdata10, ydata10)

	peak0 = f0(loc0)
	peak1 = f00(loc1)
	peak2 = f01(loc2)
	peak3 = f10(loc3)

	if drawPlot:
		nbins = 100

		fig = plt.figure()
		ax = fig.add_subplot(111)
		
		n, bins, _ = ax.hist(data, bins=nbins, alpha=0.2)
		bin_width = bins[1] - bins[0]	
		cutoff_idx = np.searchsorted(bins, cutoff)[0]
		print("idx: ", cutoff_idx)
		area0 = bin_width * sum(n[:cutoff_idx])
		area1 = bin_width * sum(n[cutoff_idx:])
		area = area0+area1
		print("integral: ", area)
		ratio0 = area0
		ratio1 = area1
		
		#ax.plot(xdata0, ratio0*ydata0, label=("FWHM: {:.0f}".format(round(width0))), linewidth=3)	
		ax.plot(xdata00, ratio0*ydata00, label=("FWHM: {:.0f}".format(round(width00))), linewidth=3)
		ax.plot(xdata01, ratio0*ydata01, label=("FWHM: {:.0f}".format(round(width01))), linewidth=3)
		ax.plot(xdata10, ratio1*ydata10, label=("FWHM: {:.0f}".format(round(width10))), linewidth=3)
		#ax.hlines(ratio0*width_height0, left_ips0, right_ips0)
		#ax.hlines(ratio0*width_height00, left_ips00, right_ips00)
		#ax.hlines(ratio0*width_height01, left_ips01, right_ips01)
		#ax.hlines(ratio1*width_height10, left_ips10, right_ips10)
		
		#ax.vlines(e_low, 0, ratio0*f0(e_low), linestyles='dashed')
		#ax.vlines(loc0, 0, ratio0*peak0, linestyles='dashed')
		ax.vlines(loc1, 0, ratio0*peak1, linestyles='dashed')
		ax.vlines(loc2, 0, ratio0*peak2, linestyles='dashed')
		ax.vlines(loc3, 0, ratio1*peak3, linestyles='dashed')

		ax.set_ylim(ymin=0)
		ax.legend(loc='upper right')
		ax.set_xlabel("Energy [Ev]")
		ax.set_ylabel("Counts")
		ax.set_title("Fe55 Energy Spectrum")

		plt.show()

def getFWHM_separatePeaks(data, npeaks=None, bw_list=None, samples=1000, desc="", xlabel="",  drawPlot=True):
	
	x = np.linspace(np.amin(data), np.amax(data), samples)
	kernel = stats.gaussian_kde(data)
	dist = kernel(x)

	peak_indices, peak_properties = find_peaks(dist, height=0)
	npeaks_total = peak_indices.size
	
	# ensure npeaks is valid
	if npeaks is None:
		npeaks = npeaks_total
	elif not isinstance(npeaks, int):
		raise ValueError("Number of peaks must be an integer value.")
	elif not npeaks > 0:
		raise ValueError("Number of peaks  must be greater than 0.")
	elif (npeaks > npeaks_total):
		print("Can only find {0} peaks in the data. Assuming {0} peaks instead of {1}.".format(npeaks_total, npeaks))
		npeaks = npeaks_total

	if bw_list is not None:
		if not (len(bw_list) == npeaks):
			raise ValueError("Bandwidth list must match number of peaks.")

	else:
		bw_list = []
		for i in range(npeaks):
			bw_list.append(None)

	# get npeaks greatest peak heights
	peak_heights = peak_properties["peak_heights"]
	idx = np.sort(np.argsort(peak_heights)[-npeaks:])
	peak_indices_filtered = np.take(peak_indices, idx)
	peak_heights_filtered = np.take(peak_heights, idx)
	rel_peak_heights = peak_heights_filtered/np.sum(peak_heights_filtered)	

	# get most prominent minimums
	cutoffs = getCutoffs(data, npeaks, samples)	
	cutoffs = np.append(cutoffs, [np.amin(data)-1, np.amax(data)+1])
	cutoffs = np.sort(cutoffs)
	
	print("Cutoffs: ", cutoffs)

	# split the data into peaks
	data_split = []
	fwhm_list = []
	dist_list = []

	for i in range(cutoffs.size - 1):
		data_split.append(data[(data>cutoffs[i]) & (data<=cutoffs[i+1])])
		fwhm, dist = getFWHM(data_split[i], npeaks=1, bw=bw_list[i], samples=samples, drawPlot=False)
		fwhm_list.append(fwhm)
		dist_list.append(dist)

	if drawPlot:
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.hist(data, bins=100, density=True)
		for i, (sdata, dist, height) in enumerate(zip(data_split, dist_list, rel_peak_heights)):
			x = np.linspace(np.amin(sdata), np.amax(sdata), samples)
			if bw_list[i] is None:
				bw_str = "Default"
			else:
				bw_str = str(bw_list[i])
			#ax.plot(x, dist*height, label="BW: " + bw_str + " FWHM: " + str(round(fwhm_list[i].item(),2)))
			ax.plot(x, dist*height, label="FWHM: " + str(round(fwhm_list[i].item())) + " eV", linewidth=3)
			ax.set_ylabel("Frequencies")
			if not xlabel=="":
				ax.set_xlabel(xlabel)

			title = "Fe55 Energies"
			if not desc=="":
				title = title + ": " + desc
			
			ax.set_title(title)

			#ax.set_xlim(5500, 6800)		

			ax.legend(loc='upper right')

			
		plt.show()

	return np.array(fwhm_list)

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
	

