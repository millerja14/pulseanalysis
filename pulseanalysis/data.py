import mkidcalculator as mc

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np

directory = "./data"

# FOR SPECIFIC DATA SET
# REMOVE OUTLIERS
outliers = []
outliers_extra_peaks = []

coords = "/loop_geometric_masked.p"
#coords = "/loop_spectra.p"
combined = True

def loadEnergies(direct=directory):

	loop = mc.Loop.from_pickle(direct + coords)
	energies = []

	for pulse in loop.pulses:
		energies.append(pulse.energies[0])

	energies.sort()
	return energies

def loadTraces(direct=directory):

	'''
	Load pulse tracs from KID data given a data directory.
	'''

	loop = mc.Loop.from_pickle(direct + coords)

	plength = loop.pulses[0].p_trace[0].size
	dlength = loop.pulses[0].d_trace[0].size

	ptraces = np.zeros((0,plength))
	dtraces = np.zeros((0,dlength))

	for i, pulse in enumerate(loop.pulses):
		ptraces = np.append(ptraces, [pulse.p_trace[pulse.mask]], axis=0)
		dtraces = np.append(dtraces, [pulse.d_trace[pulse.mask]], axis=0)

	traces = ptraces
	if combined:
		traces = np.concatenate((ptraces, dtraces), axis=1)

	toRemove = np.union1d(outliers, outliers_extra_peaks)

	traces = np.delete(traces, toRemove, axis=0)

	return traces

def loadTraces_labeled(direct=directory):
	loop = mc.Loop.from_pickle(direct + coords)
	loop._set_directory('/data/jmiller/optical_analysis/data/')

	plength = loop.pulses[0].p_trace[0].size
	dlength = loop.pulses[0].d_trace[0].size

	ptraces = np.zeros((0,plength))
	dtraces = np.zeros((0,dlength))

	counts = np.array([])
	energies = np.array([])
	ptrace_blocks = np.array([], dtype=object)
	dtrace_blocks = np.array([], dtype=object)

	for i, pulse in enumerate(loop.pulses):

		e = pulse.energies[0]
		pos = int(np.searchsorted(energies, e))
		energies = np.insert(energies, pos, e)

		# insert new data according to sorted energy
		ptraces = np.insert(ptraces, int(np.sum(counts[:pos])), pulse.p_trace[pulse.mask], axis=0)
		dtraces = np.insert(dtraces, int(np.sum(counts[:pos])), pulse.d_trace[pulse.mask], axis=0)

		count = pulse.p_trace[pulse.mask].shape[0]
		print("Energy: {:.2f}eV Count: {}".format(e, count))
		counts = np.insert(counts, pos, count)

	#ptraces = np.concatenate(tuple(ptrace_blocks), axis=0)
	#dtraces = np.concatenate(tuple(dtrace_blocks), axis=0)

	# construct label array
	labels = np.array([])
	for label, count in enumerate(counts):
		labels = np.append(labels, np.ones(int(count))*label)

	traces = ptraces
	if combined:
		traces = np.concatenate((ptraces, dtraces), axis=1)

	#toRemove = np.union1d(outliers, outliers_extra_peaks)

	#traces = np.delete(traces, toRemove, axis=0)

	print("M: ", len(traces[0]))

	return traces, labels

def loadTraces_split(s=0.5, seed=None, direct=directory):
	if s > 1 or s < 0:
		return ValueError("S must be between 0 and 1")

	traces = loadTraces(direct=direct)

	num = len(traces)
	indices = np.arange(num)

	if seed is not None:
		np.random.seed(seed)
		np.random.shuffle(indices)

	split = int(num*s)

	indices1 = indices[:split]
	indices2 = indices[split:]

	traces1 = np.take(traces, indices1, axis=0)
	traces2 = np.take(traces, indices2, axis=0)

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
