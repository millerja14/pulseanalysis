import mkidcalculator as mc

directory = "./data"

def loadTraces(dir=directory):

	'''
	Load pulse tracs from KID data given a data directory.
	'''
	
	loop = mc.Loop.from_pickle(directory + "/analysis/loop_combined.p")
	traces = loop.pulses[0].p_trace

	return traces
