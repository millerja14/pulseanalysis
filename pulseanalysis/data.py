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

#coords = "/loop_geometric_masked.p"
coords = "/loop_analytic_masked.p"
combined = True

def loadEnergies(direct=directory):
	
	loop = mc.Loop.from_pickle(direct + coords)
	energies = []

	for pulse in loop.pulses:
		energies.append(pulse.energies[0])

	return energies

def loadTraces(direct=directory):

	'''
	Load pulse tracs from KID data given a data directory.
	'''
	
	loop = mc.Loop.from_pickle(direct + coords)

	ptraces = np.zeros((0,2500))
	dtraces = np.zeros((0,2500))

	for i, pulse in enumerate(loop.pulses):
		ptraces = np.append(ptraces, pulse.p_trace[pulse.mask], axis=0)
		dtraces = np.append(dtraces, pulse.d_trace[pulse.mask], axis=0)

	traces = ptraces
	if combined:
		traces = np.concatenate((ptraces, dtraces), axis=1)

	toRemove = np.union1d(outliers, outliers_extra_peaks)

	traces = np.delete(traces, toRemove, axis=0)	

	return traces

def loadTraces_labeled(direct=directory):
	
	loop = mc.Loop.from_pickle(direct + coords)
	
	ptraces = np.zeros((0,2500))
	dtraces = np.zeros((0,2500))
	
	labels = np.array([])

	for i, pulse in enumerate(loop.pulses):

		ptraces = np.append(ptraces, pulse.p_trace[pulse.mask], axis=0)
		dtraces = np.append(dtraces, pulse.d_trace[pulse.mask], axis=0)

		num = pulse.p_trace[pulse.mask].shape[0]
		labels = np.append(labels, np.ones(num)*i)
	
	traces = ptraces
	if combined:
		traces = np.concatenate((ptraces, dtraces), axis=1)

	toRemove = np.union1d(outliers, outliers_extra_peaks)

	traces = np.delete(traces, toRemove, axis=0)

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
	
