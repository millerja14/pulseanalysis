import mkidcalculator as mc

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np

directory = "./data"

def loadTraces(dir=directory):

	'''
	Load pulse tracs from KID data given a data directory.
	'''
	
	loop = mc.Loop.from_pickle(directory + "/analysis/loop_combined.p")
	traces = loop.pulses[0].p_trace

	return traces

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
	ax.set_title("Photon Pulse")

	plt.show()
	
