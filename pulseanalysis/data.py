import mkidcalculator as mc

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np

directory = "./data"

def loadTraces(direct=directory):

	'''
	Load pulse tracs from KID data given a data directory.
	'''
	
	loop = mc.Loop.from_pickle(direct + "/analysis/loop_combined.p")
	ptraces = loop.pulses[0].p_trace
	dtraces = loop.pulses[0].d_trace	
	traces = ptraces
	#traces = np.concatenate((ptraces, dtraces), axis=1)

	# FOR SPECIFIC DATA SET
	# REMOVE OUTLIERS
	outliers = [3810]
	outliers_extra_peaks = [1033, 1035, 1061, 1163,  118, 1247, 1248,  126, 1264, 1266, 1339,
	1353, 1354, 1360, 1387,  141, 1410, 1415, 1416, 1419, 1422, 1518,
	1524, 1553, 1569, 1613, 1616, 1619, 1649, 1661, 1714, 1716, 1764,
	1809, 1856, 1862, 1881, 1884, 1892, 1897, 1900, 1954, 1960, 1995,
	2006, 2007,  203, 2036, 2040, 2047, 2068, 2070, 2077, 2078, 2089,
	2090, 2109, 2114, 2133, 2134, 2138, 2139, 2141, 2148, 2175, 2176,
	2194, 2197, 2211, 2212, 2217, 2225, 2237, 2240, 2244, 2248, 2251,
	2264, 2268, 2294, 2322, 2326, 2327, 2328, 2333, 2362, 2373, 2382,
	2392, 2397, 2401, 2413, 2414, 2420, 2435, 2438, 2443, 2448, 2450,
	2481, 2488, 2513, 2530, 2550, 2564, 2570, 2601, 2606, 2633, 2652,
	2653, 2677, 2684, 2701, 2718, 2725, 2740, 2741, 2743, 2768, 2774,
	2794, 2795, 2821, 2833, 2869, 2876, 2895, 2900, 2915, 2919, 2932,
	294, 2945, 2947, 2956, 2985, 3018, 3035, 3048, 3051, 3052, 3059,
	3061, 3068, 3087, 3105, 3122, 3129, 3135, 3136, 3150, 3157, 3159,
	3173, 3176,  318, 3184, 3186, 3193, 3194, 3198, 3205, 3212, 3215,
	3226, 3232, 3236, 3252, 3276, 3279, 3295, 3298, 3299, 3314, 3315,
	3329, 3332, 3333, 3339, 3342, 3351, 3356, 3365, 3373, 3380, 3387,
	3406, 3422, 3424, 3443, 3444, 3462, 3465, 3472, 3474, 3502, 3507,
	3517, 3527, 3531, 3542, 3552, 3576,  358, 3588, 3596, 3597,  360,
	3614, 3615, 3635, 3636, 3639, 3642, 3655, 3672, 3705,  371, 3716,
	3722, 3723, 3727, 3731, 3738, 3748, 3765, 3770, 3776, 3780, 3784,
	3787, 3810, 3815,  382,  384,   39,  424,  436,   44,  489,   53,
	57,  589,  620,  630,  631,  639,  684,  688,    7,  701,  757,
	765,  819,  826,  827,  837,  872,  898,    9,  907,  941,  985,
	991]
	#toRemove = []

	toRemove = np.union1d(outliers, outliers_extra_peaks)

	traces = np.delete(traces, toRemove, axis=0)	

	return traces

def loadTraces_split(s=0.5, direct=directory):
	if s > 1 or s < 0:
		return ValueError("S must be between 0 and 1")

	traces = loadTraces(direct=direct)
	
	num = len(traces)
	split = int(num*s)

	traces1 = traces[:split]
	traces2 = traces[split:]

	return traces1, traces2

def plotTrace(pulse=None):
	if not hasattr(pulse, 'p_trace'):
		loop = mc.Loop.from_pickle(directory + "/analysis/loop_combined.p")
		pulse = loop.pulses[0]
		trace = pulse.p_trace[0]
		rate = pulse.sample_rate
	
	x = np.arange(0, trace.size)*(1/rate)*(10**3)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(x, trace)
	ax.set_xlabel("Time [mS]")
	ax.set_ylabel("Phase Shift [rad]")
	ax.set_title("Fe55 Photon Pulse")

	plt.show()
	
