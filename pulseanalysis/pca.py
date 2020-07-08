import mkidcalculator as mc
import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

directory = "./data"

def loadTraces(dir=directory):
	
	'''
	Load pulse traces from MKID data at default directory.
	'''

	loop = mc.Loop.from_pickle(directory + "/analysis/loop_combined.p")
	traces = loop.pulses[0].p_trace

	return traces

def generate2DScatter(traces=None, drawPlot=False):
	
	'''
	Given a 2D collection of data where the first dimension contains
	each vector, generate a 2-component pca and return coordinates of each
	vector in this space. Set drawPlot to False to supress plot.
	'''

	if traces==None:
		traces = loadTraces()

	nPoints = traces.shape[0]

	traceAvg = np.mean(traces, axis=0)

	B = traces - np.tile(traceAvg, (nPoints,1))

	U, S, VT = np.linalg.svd(B/np.sqrt(nPoints), full_matrices=0)

	points = np.zeros(shape=(nPoints, 2))

	for j in range(B.shape[0]):
		x = VT[0,:] @ B[j,:].T
		y = VT[1,:] @ B[j,:].T
		points[j,:] = [x,y]	

	if drawPlot:
		fig = plt.figure()
		ax = fig.add_subplot(111)

		ax.scatter(points[:,0], points[:,1], marker='x', color='b')	
	
		ax.set_title("PCA of Fe55 Data")
		ax.set_xlabel("PC1")
		ax.set_ylabel("PC2")

		plt.show()

	return points

def project2DScatter(points=None, direction=[8,5], drawPlot=False):
	
	'''
	Given an nx2 array of points, project these points on to the
	direction that is given as a vector of size 2. Returns array of size n
	containing 1D data.
	'''

	if points==None:
		points = generate2DScatter()

	sym = np.array(direction)
	unit_sym = sym/np.linalg.norm(sym)
	proj = points @ unit_sym

	if drawPlot:
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.hist(proj, bins='auto')
		
		ax.set_title("2D PCA projected on to <{0:.2f},{1:.2f}>".format(direction[0], direction[1]))

		plt.show()

	return proj

#fig1 = plt.figure()
#ax1 = fig1.add_subplot(121)
#ax1.semilogy(S, '-o', color='k')
#ax2 = fig1.add_subplot(122)
#ax2.plot(np.cumsum(S)/np.sum(S), '-o', color='k')
#plt.show()

