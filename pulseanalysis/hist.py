import mkidcalculator as mc
import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.signal import find_peaks, peak_widths
from scipy import interpolate, optimize

import pulseanalysis.data as mkid

import matplotlib.animation as animation
from matplotlib.ticker import FormatStrFormatter

import cmath

A=8.2
B=16.2
C=2.85

loc1 = 5887.65
loc2 = 5898.75
loc3 = 6490.45
loc0 = (A*loc1+B*loc2)/(A+B)

e_low = loc0
e_high = loc3
e_cutoff = 6250
e_peaks = np.array([e_low, e_high])

def S21(f, fr, Qc=2*10**5, Qi=2*10**5):
        Q = 1 / ((1/Qc) + (1/Qi))
        S = 1 - (Q/Qc) / ( 1 + 2j * Q * (f-fr)/(fr) )
        return 20*np.log10(abs(S))

def transfer(f, L):

	if f == 0:
		return np.pi/2

	w = 2*np.pi*f

	R = 10**2
	C1 = 10**-3
	C2 = 10**-7

	IR = R
	#IC1 = 1 / (1j*w*C1)
	IC1 = 0
	IC2 = 1 / (1j*w*C2)
	IL = 1j*w*L

	I_p = 1/((1/IL) + (1/IC2))

	I_f = IC1 + I_p

	I = I_f / (IR + I_f)

	return cmath.phase(I)

def findRoot():
	L1 = 10**-4
	L2 = L1 * 1.5

	tfunc = np.vectorize(transfer)
	f_1 = optimize.root_scalar(tfunc, args=(L1), bracket=[0.001, 10**5], method="brentq").root
	f_2 = optimize.root_scalar(tfunc, args=(L2), bracket=[0.001, 10**5], method="brentq").root
	x_range = np.linspace(0, 2*f_1, 10**3)
	y1_range = tfunc(x_range, L1)
	y2_range = tfunc(x_range, L2)
	fig = plt.figure()
	ax = fig.add_subplot()
	ax.plot(x_range, y1_range+np.pi, lw=3, label="before incidence")
	ax.plot(x_range, y2_range+np.pi, lw=3, linestyle="dashed", label="on incidence")
	ax.set_xticks([f_1])
	ax.set_xticklabels([r"$f_0$"])
	ax.axvline(x=f_1, color="black", lw=2)
	ax.axhline(y=tfunc(f_1, L1)+np.pi, xmin=0.1, xmax=0.3, color="black", lw=2)
	ax.axhline(y=tfunc(f_1, L2)+np.pi, xmin=0.1, xmax=0.3, color="black", lw=2)
	ax.set_xlabel("Frequency")
	ax.set_ylabel("Phase Shift [rad]")
	ax.set_title("KID Photon Response")
	ax.legend(loc="upper right")
	plt.show()

def readoutAnimation(samples=1000):
	fr1 = 5
	f_spread = 0.001/2
	fr2 = fr1 - f_spread/8

	f_array = np.linspace(fr1-f_spread, fr1+f_spread, samples)
	s1_array = S21(f_array, fr1, Qc=1*10**5, Qi=1*10**5)
	s2_array = 0.5 * S21(f_array, fr2, Qc=4*10**4, Qi=4*10**4)

	f1 = interpolate.interp1d(f_array, s1_array)
	f2 = interpolate.interp1d(f_array, s2_array)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	#ax.set_ylim(-0.9, 0.1)
	#ax.set_xlim(fr1-f_spread, fr1+f_spread)
	ax.set_xticks([fr2, fr1])
	ax.set_xticklabels([r"$f'$", r"$f_0$"])
	#ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	ax.plot(f_array, s1_array, lw=3, label="before incidence")
	ax.plot(f_array, s2_array, linestyle='dashed', lw=3, label="on incidence")
	ax.axvline(x=fr1, color="black", lw=2)
	ax.axvline(x=fr2, color="black", linestyle="dashed", lw=2)
	ax.axhline(y=f1(fr1), xmin=0.6, xmax=0.8, color="black", lw=2)
	ax.axhline(y=f2(fr1), xmin=0.6, xmax=0.8, color="black", lw=2)
	ax.set_ylabel("Power [dB]")
	ax.set_xlabel("Frequency")
	ax.set_title("KID Photon Response")
	ax.legend(loc="lower left")
	plt.show()

def plotDetectorResponse(samples=1000):
	fig = plt.figure()
	fig.set_size_inches(27, 10.8, True)
	ax1 = fig.add_subplot(121)
	ax2 = fig.add_subplot(122)

	fr1 = 5
	f_spread = 0.001/2
	fr2 = fr1 - f_spread/8

	f_array = np.linspace(fr1-f_spread, fr1+f_spread, samples)
	s1_array = S21(f_array, fr1, Qc=1*10**5, Qi=1*10**5)
	s2_array = 0.5 * S21(f_array, fr2, Qc=4*10**4, Qi=4*10**4)

	f1 = interpolate.interp1d(f_array, s1_array)
	f2 = interpolate.interp1d(f_array, s2_array)

	#ax.set_ylim(-0.9, 0.1)
	#ax.set_xlim(fr1-f_spread, fr1+f_spread)
	ax1.set_xticks([fr2, fr1])
	ax1.set_xticklabels([r"${f_r}'$", r"$f_r$"])
	#ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	ax1.plot(f_array, s1_array, lw=3, label="before incidence")
	ax1.plot(f_array, s2_array, linestyle='dashed', lw=3, label="on incidence")
	ax1.axvline(x=fr1, color="black", lw=2)
	ax1.axvline(x=fr2, color="black", linestyle="dashed", lw=2)
	ax1.axhline(y=f1(fr1), xmin=0.5, xmax=0.8, color="gray", lw=2)
	ax1.axhline(y=f2(fr1), xmin=0.5, xmax=0.8, color="gray", lw=2)
	ax1.set_ylabel("Power [dB]")
	ax1.set_xlabel("Frequency")
	ax1.set_title("Dissipation")
	#ax1.legend(loc="lower left")

	L1 = 10**-4
	L2 = L1 * 1.5

	tfunc = np.vectorize(transfer)
	f_1 = optimize.root_scalar(tfunc, args=(L1), bracket=[0.001, 10**5], method="brentq").root
	f_2 = optimize.root_scalar(tfunc, args=(L2), bracket=[0.001, 10**5], method="brentq").root
	x_range = np.linspace(0, 2*f_1, samples)
	y1_range = tfunc(x_range, L1)
	y2_range = tfunc(x_range, L2)

	ax2.plot(x_range, y1_range+np.pi, lw=3, label="before incidence")
	ax2.plot(x_range, y2_range+np.pi, lw=3, linestyle="dashed", label="on incidence")
	ax2.set_xticks([f_2, f_1])
	ax2.set_xticklabels([r"${f_r}'$", r"$f_r$"])
	ax2.axvline(x=f_1, color="black", lw=2)
	ax2.axvline(x=f_2, color="black", lw=2, linestyle="dashed")
	ax2.axhline(y=tfunc(f_1, L1)+np.pi, xmin=0.1, xmax=0.5, color="gray", lw=2)
	ax2.axhline(y=tfunc(f_1, L2)+np.pi, xmin=0.1, xmax=0.5, color="gray", lw=2)
	ax2.set_xlabel("Frequency")
	ax2.set_ylabel("Phase Shift [rad]")
	ax2.set_title("Phase Shift")
	ax2.legend(loc="upper right")

	fig.suptitle("KID Photon Response")

	plt.show()

def fe55_distribution(x_array, fwhm, x_peaks=[loc1, loc2, loc3], y_peaks=[A, B, C]):
	sigma = fwhm/2.35482

	x_peaks = np.array(x_peaks)
	y_peaks = np.array(y_peaks)

	y_peaks_rel = y_peaks/(np.sum(y_peaks))

	fx_array = np.zeros_like(x_array)
	for loc, height in zip(x_peaks, y_peaks_rel):
		fx_array += height * stats.norm.pdf(x_array, loc, sigma)

	return fx_array

def fe55_distribution_animate(fwhm_min, fwhm_max, xmin=5800, xmax=6600, samples=3000):
	height_scale = 1.6 * B/(A+B+C)

	fig = plt.figure()
	fig.set_size_inches(19.2, 10.8, True)
	#ax = plt.axes(xlabel="Energy [eV]", ylabel="Probability Density", xlim=(xmin, xmax), ylim=(0, height_scale * stats.norm.pdf(loc2, loc2, fwhm_min/2.35482)))
	ax = plt.axes(xlim=(xmin, xmax))
	ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	line, = ax.plot([], [], lw=3)
	x_array = np.linspace(xmin, xmax, samples)
	x_array = np.append(x_array, [loc1, loc2, loc3])
	x_array = np.sort(x_array)

	def init():
		line.set_data([], [])
		return line,

	def animate(i):
		#x_array = np.linspace(xmin, xmax, samples)
		fx_array = fe55_distribution(x_array, i)
		line.set_data(x_array, fx_array)
		line.set_label(r"$\Delta$" + "E: {:.0f} eV".format(round(i)))
		ax.legend(loc="upper right")
		ax.set_ylim(0, height_scale * stats.norm.pdf(loc2, loc2, i/2.35482))
		ax.set_ylabel("Probability Density")
		ax.set_xlabel("Energy [eV]")
		ax.set_title("Fe55 Radioactive Spectrum")
		return line,

	dsteps_up = np.linspace(fwhm_min, fwhm_max, 150)
	#dsteps_down = np.linspace(fwhm_max, fwhm_min, 150)
	#dsteps = np.concatenate((dsteps_up, dsteps_down))
	dsteps = dsteps_up

	anim = animation.FuncAnimation(fig, animate, init_func=init, frames=dsteps, interval=100, blit=True, repeat=False)

	writer = animation.PillowWriter(fps=30)
	anim.save("./fe55spectrum.gif", writer=writer, dpi=50)

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

	if drawPlot:
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

def distToEV(values, peaks=e_peaks):
	cutoff = getCutoffs(values, 2)
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

def getFWHM_fe55(data=None, x0=loc1, x1=loc2, A=A, B=B, loops=20, bw_list=[.17,.22], samples=10000, id=""):

    """
    Takes in array of energies for Iron-55 data and outputs fitted histogram with FWHM displayed. Left peak data
    is resolved into two peaks.
    """

    # normalize relative heights
    total = A+B
    A = A/total
    B = B/total

    # load benchmark data if no data is given
    if data is None:
    	data = benchmarkEnergies()

    # split the data into left peak and right peak data
    cutoff = getCutoffs(data, 2)
    data0 = data[data<cutoff]
    data1 = data[data>=cutoff]

    # resolve left peak as if it is a single peak and compute information
    xdata0, ydata0 = resolveSinglePeak(data0, bw=bw_list[0], samples=samples)
    width0, width_height0, left_ips0, right_ips0 = fwhmFromPeak(xdata0, ydata0)

    # resolve double peak template and store as x and y data
    xdata0x, ydata0x = resolveDoublePeak(data0, x0=x0, x1=x1, A=A, B=B, bw=bw_list[0], loops=loops, drawPlot=False)

    # generate data for double peak using double peak template
    xdata00 = xdata0x + x0
    xdata01 = xdata0x + x1
    ydata00 = A*ydata0x
    ydata01 = B*ydata0x

    # compute information on each double peak
    width00, width_height00, left_ips00, right_ips00 = fwhmFromPeak(xdata00, ydata00)
    width01, width_height01, left_ips01, right_ips01 = fwhmFromPeak(xdata01, ydata01)

    # resolve right peak and compute information
    xdata10, ydata10 = resolveSinglePeak(data1, bw=bw_list[1], samples=samples)
    width10, width_height10, left_ips10, right_ips10 = fwhmFromPeak(xdata10, ydata10)

    # generate interpolations from each peak's data
    f0 = interpolate.interp1d(xdata0, ydata0, bounds_error=False, fill_value=0)
    f00 = interpolate.interp1d(xdata00, ydata00, bounds_error=False, fill_value=0)
    f01 = interpolate.interp1d(xdata01, ydata01, bounds_error=False, fill_value=0)
    f10 = interpolate.interp1d(xdata10, ydata10, bounds_error=False, fill_value=0)

    # compute peak height at each energy location
    peak0 = f0(loc0)
    peak1 = f00(loc1)
    peak2 = f01(loc2)
    peak3 = f10(loc3)

    # generate evenly spaced x values covering the domain of the data
    xdata_threepeak = np.concatenate((xdata00, xdata01, xdata10))
    xmin = np.amin(xdata_threepeak)
    xmax = np.amax(xdata_threepeak)
    xdata = np.linspace(xmin, xmax, 5*samples)

    # number of total bins for histogram
    nbins = 100

    # start plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)

    n, bins, _ = ax.hist(data, bins=nbins, alpha=0.3)

    # compute areas of the histogram to the left and right of the cutoff
    # these values are for scaling the fit curve to match the size of the histogram
    # depending on the bin width of the histogram
    bin_width = bins[1] - bins[0]
    cutoff_idx = np.searchsorted(bins, cutoff)[0]
    area0 = bin_width * sum(n[:cutoff_idx])
    area1 = bin_width * sum(n[cutoff_idx:])
    area = area0+area1
    ratio0 = area0
    ratio1 = area1

    # function to return total fit
    f = lambda x: ratio0*f00(x) + ratio0*f01(x) + ratio1*f10(x)

    # plot and label each of three peaks and the total distribution fit
    ax.plot(xdata00, ratio0*ydata00, label=(r"$\Delta$E: {:.0f} eV".format(round(width00))), linewidth=1)
    ax.plot(xdata01, ratio0*ydata01, label=(r"$\Delta$E: {:.0f} eV".format(round(width01))), linewidth=1)
    ax.plot(xdata10, ratio1*ydata10, label=(r"$\Delta$E: {:.0f} eV".format(round(width10))), linewidth=1)
    ax.plot(xdata, f(xdata), label=("total"), linestyle=(0, (2, 1)), linewidth=1)

    # mark each peak location
    ax.vlines(loc1, 0, ratio0*peak1, linestyles='dashed')
    ax.vlines(loc2, 0, ratio0*peak2, linestyles='dashed')
    ax.vlines(loc3, 0, ratio1*peak3, linestyles='dashed')

    # plot settings
    ax.set_ylim(0, 700)
    ax.set_xlim(5750, 6600)
    ax.legend(loc='upper right', frameon=False)
    ax.set_xlabel("energy [eV]")
    ax.set_ylabel("counts per bin width")
    #ax.set_title("Iron-55 Detector Spectrum using PCA in 80D")

    # save plot
    fig.set_size_inches(3.5, 3.5)
    plt.savefig("./tkid_results{}.pdf".format(id), bbox_inches='tight')
    plt.savefig("./tkid_results{}.png".format(id), bbox_inches='tight')

    plt.close()

def getFWHM_separatePeaks(data, npeaks=None, bw_list=[.15,.2], samples=1000, desc="", xlabel="Energy [eV]",  drawPlot=True):

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
    	n, bins, _ = ax.hist(data, bins=100, alpha=0.3)
    	bin_width = bins[1] - bins[0]
    	area = sum(n)*bin_width
    	for i, (sdata, dist, height) in enumerate(zip(data_split, dist_list, rel_peak_heights)):
    		x = np.linspace(np.amin(sdata), np.amax(sdata), samples)
    		if bw_list[i] is None:
    			bw_str = "Default"
    		else:
    			bw_str = str(bw_list[i])
    		#ax.plot(x, dist*height, label="BW: " + bw_str + " FWHM: " + str(round(fwhm_list[i].item(),2)))
    		ax.plot(x, dist*height*area, label=r"$\Delta$" + "E: " + str(round(fwhm_list[i].item())) + " eV", linewidth=3)
    		ax.set_ylabel("Counts")
    		if not xlabel=="":
    			ax.set_xlabel(xlabel)

    		title = r"${}^{55}$" + "Fe Energy Spectrum"
    		if not desc=="":
    			title = title + ": " + desc

    		#ax.set_title(title)

    		ax.set_xlim(5650, 6650)

    		ax.legend(loc='upper right')


    	plt.show()

    return np.array(fwhm_list).flatten()

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
