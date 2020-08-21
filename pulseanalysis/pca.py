cores="8"

import os
os.environ["MKL_NUM_THREADS"] = cores
os.environ["NUMEXPR_NUM_THREADS"] = cores 
os.environ["OMP_NUM_THREADS"] = cores

import pickle
import mkidcalculator as mc
import numpy as np

import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
mpl.rcParams['font.size'] = 22
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['axes.labelpad'] = 6.0
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

import scipy.optimize as optimize
import scipy.spatial.transform as transform

import pulseanalysis.hist as hist
import pulseanalysis.data as mkid

db_path = "./pca_data/optimization.pickle"

e_high = 6490
e_low = 5900

def plotNComponents(n, label, traces=None):
	if not isinstance(traces, np.ndarray):
		print("No traces given, getting default traces...")
		traces = mkid.loadTraces()
	
	if not isinstance(label, str):
		raise ValueError("Label must be a string")
	
	comp_path = "./decomp/" + label + "/"

	if not os.path.exists(comp_path):
		os.makedirs(comp_path)

	nPoints = traces.shape[0]

	traceAvg = np.mean(traces, axis=0)

	B = traces - np.tile(traceAvg, (nPoints, 1))

	U, S, VT = np.linalg.svd(B, full_matrices=False)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(traces[0])
	ax.set_title("Full Pulse")
	ax.axes.xaxis.set_visible(False)
	ax.axes.yaxis.set_visible(False)
	plt.savefig(comp_path + "pulse.png")
	plt.close()
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(traceAvg)
	ax.set_title("Average Pulse")
	ax.axes.xaxis.set_visible(False)
	ax.axes.yaxis.set_visible(False)
	plt.savefig(comp_path + "PC0.png")
	plt.close()
	
	print(S)
	varfrac = 100*(S**2/np.sum(S**2))

	for i in range(n):
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot(-VT[i,:])
		ax.axes.xaxis.set_visible(False)
		ax.axes.yaxis.set_visible(False)
		ax.set_title("PC{0}: {1:.2f}% of Variance".format(i+1, varfrac[i]))
		plt.savefig(comp_path + "PC{0}.png".format(i+1))
		plt.close()

def saveAllTrace(traces=None):
	if not isinstance(traces, np.ndarray):
		print("No traces given, getting default traces...")
		traces = mkid.loadTraces()

	for i, trace in enumerate(traces):
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot(trace)
		ax.set_title("Trace Index: " + str(i))
		ax.axes.xaxis.set_visible(False)
		ax.axes.yaxis.set_visible(False)
		plt.savefig("./traces/trace{}.png".format(i))
		plt.close()

def plotTrace(comp_list, weights, traces=None):
	comp_list = np.array(comp_list)
	weights = np.array(weights)
	weights = weights/np.linalg.norm(weights)
	
	if comp_list.size != weights.size:
		raise ValueError("Weights array must have same size as component list")
	
	if not isinstance(traces, np.ndarray):
		print("No traces given, getting default traces...")
		traces = mkid.loadTraces()
	
	nPoints = traces.shape[0]
	
	traceAvg = np.mean(traces, axis=0)

	B = traces - np.tile(traceAvg, (nPoints, 1))

	U, S, VT = np.linalg.svd(B, full_matrices=False)

	result = np.zeros(traceAvg.size)

	for comp, weight in zip(comp_list, weights):
		comp_trace = VT[comp-1,:]
		result += comp_trace*weight

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(result)
	ax.axes.xaxis.set_visible(False)
	ax.axes.yaxis.set_visible(False)
	ax.set_title("Comps : " + str(comp_list))
	plt.show()
		
def principalVariances(traces=None):
	if not isinstance(traces, np.ndarray):
		print("No traces given, getting default traces...")
		traces = mkid.loadTraces()

	nPoints = traces.shape[0]

	traceAvg = np.mean(traces, axis=0)

	B = traces - np.tile(traceAvg, (nPoints,1))

	U, S, VT = np.linalg.svd(B, full_matrices=False)

	varfrac = np.cumsum(S**2)/np.sum(S**2)

	fig = plt.figure()
	ax = fig.add_subplot(111)

	ax.plot(varfrac, '-o', color='k')

	ax.set_xlabel("PC Number")
	ax.set_ylabel("Cumulative Proportion of Variance")

	ax.set_title("Variance in Principal Components")

	plt.show()

def getPCABasis(traces=None):

	if not isinstance(traces, (np.ndarray)):
		print("generate2DScatter(): No traces given, getting default traces...")
		traces = mkid.loadTraces()

	nPoints = traces.shape[0]

	traceAvg = np.mean(traces, axis=0)

	B = traces - np.tile(traceAvg, (nPoints,1))

	_, _, VT = np.linalg.svd(B/np.sqrt(nPoints), full_matrices=0)

	return VT

def generate2DScatter(traces=None, drawPlot=False):
	
	'''
	Given a 2D collection of data where the first dimension contains
	each vector, generate a 2-component pca and return coordinates of each
	vector in this space. Set drawPlot to False to supress plot.
	'''

	if not isinstance(traces, (np.ndarray)):
		print("generate2DScatter(): No traces given, getting default traces...")
		traces = mkid.loadTraces()

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

def generate3DScatter(traces=None, drawPlot=True):
	if not isinstance(traces, (np.ndarray)):
		print("generate3DScatter(): No traces given, getting default traces...")
		traces = mkid.loadTraces()

	nPoints = traces.shape[0]
	traceAvg = np.mean(traces, axis=0)

	B = traces - np.tile(traceAvg, (nPoints,1))

	U, S, VT = np.linalg.svd(B/np.sqrt(nPoints), full_matrices=0)

	points = np.zeros(shape=(nPoints, 3))

	for j in range(B.shape[0]):
		x = VT[0,:] @ B[j,:].T
		y = VT[1,:] @ B[j,:].T
		z = VT[2,:] @ B[j,:].T
		points[j,:] = [x,y,z]

	if drawPlot:
		fig = plt.figure()
		ax = plt.axes(projection='3d')

		ax.scatter(points[:,0], points[:,1], points[:,2], marker='x', color='b')

		ax.set_title("PCA of Fe55 Data")
		ax.set_xlabel("PC1")
		ax.set_ylabel("PC2")
		ax.set_zlabel("PC3")

		plt.show()

	return points

def generateLabels(traces):
	energies = hist.benchmarkEnergies(traces)
	cutoff = hist.getCutoffs(energies, 2)

	labels = np.zeros(energies.size)
	for i, e in enumerate(energies):
		if e > cutoff:
			labels[i] = 1

	return labels

def generateScatter(dim, traces=None):
	if not isinstance(traces, np.ndarray):
		print("No traces given, getting default traces...")
		traces = mkid.loadTraces()

	if not isinstance(dim, int):
		raise ValueError("Dimension must be an integer.")

	if not (dim > 0):
		raise ValueError("Dimension must be greater than zero.")	

	nPoints = traces.shape[0]
	traceAvg = np.mean(traces, axis=0)

	B = traces - np.tile(traceAvg, (nPoints, 1))

	U, S, VT = np.linalg.svd(B/np.sqrt(nPoints), full_matrices=0)
	
	points = np.zeros(shape=(nPoints, dim))

	for j in range(B.shape[0]):
		points[j,:] = VT[:dim,:] @ B[j,:].T

	return points

def generateScatter_labeled(dim, traces=None):
	if not isinstance(traces, np.ndarray):
		print("No traces given, getting default traces...")
		traces = mkid.loadTraces()
	
	if not isinstance(dim, int):
		 raise ValueError("Dimension must be an integer.")
	
	if not (dim > 0):
		raise ValueError("Dimension must be greater than zero.")

	points = generateScatter(dim=dim, traces=traces)
	labels = generateLabels(traces)

	return points, labels

def generateScatter_labeled_nthComps(comp_list=[1,2,3,4,9,15], traces=None):
	if not isinstance(traces, np.ndarray):
		print("No traces given, getting default traces...")
		traces = mkid.loadTraces()

	labels = generateLabels(traces)
	
	comp_list = np.array(comp_list)
	dim = np.size(comp_list)

	nPoints = traces.shape[0]
	traceAvg = np.mean(traces, axis=0)

	B = traces-np.tile(traceAvg, (nPoints, 1))
	
	U, S, VT = np.linalg.svd(B/np.sqrt(nPoints), full_matrices=0)

	points = np.zeros(shape=(nPoints, dim))

	for j in range(B.shape[0]):
		points[j,:] = np.take(VT, (comp_list-1), axis=0) @ B[j,:].T

	return points, labels

def generateScatter3D_labeled(traces=None):
	if not isinstance(traces, np.ndarray):
		print("No traces given, getting default traces...")
		traces = mkid.loadTraces()

	points, labels = generateScatter_labeled(3, traces=traces)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	for i, p in enumerate(points):
		if labels[i] == 0:
			ax.scatter(*p, marker='x', color='r', s=50)
		else:
			ax.scatter(*p, marker='o', color='b', s=50)

	plt.show()

	return points, labels

def project2DScatter(points=None, direction=[8,5], drawPlot=False):
	
	'''
	Given an nx2 array of points, project these points on to the
	direction that is given as a vector of size 2. Returns array of size n
	containing 1D data.
	'''

	if not isinstance(points, (list, np.ndarray)):
		print("project2DScatter(): No points given, getting default points...")
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

def project3DScatter(points=None, direction=[8,5,0], drawPlot=False):
	if not isinstance(points, (list, np.ndarray)):
		print("project3DScatter(): No points given, getting default points...")
		points = generate2DScatter()

	sym = np.array(direction)
	unit_sym = sym/np.linalg.norm(sym)
	proj = points @ unit_sym

	if drawPlot:
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.hist(proj, bins='auto')

		ax.set_title("3D PCA projected on to <{0:.2f},{1:.2f}, {3:.2f}>".format(*direction))

		plt.show()

	return proj

def projectScatter(direction, points=None, drawPlot=False):
	if not isinstance(points, (list, np.ndarray)):
		print("project3DScatter(): No points given, getting default points...")
		points = generate2DScatter()

	if not points.shape[1] == direction.size:
		raise ValueError("Dimension of points and projection vector must match.")

	dim = direction.size

	sym = np.array(direction)
	unit_sym = sym/np.linalg.norm(sym)
	proj = points @ unit_sym

	if drawPlot:
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.hist(proj, bins='auto')

		ax.set_title(str(dim) + "D PCA projection")

		plt.show()

	return proj

def distToEV_withLabels(data, labels):
	
	#scale data
	peak_sep_ev = e_high - e_low		

	data0 = data[labels == 0]
	data1 = data[labels == 1]

	pos0 = np.mean(data0)
	pos1 = np.mean(data1)

	deltapeak = np.abs(pos0-pos1)

	scale = peak_sep_ev/deltapeak

	if pos0 > pos1:
		scale = -scale

	data_scaled = np.array(data*scale)

	#shift data
	peak0 = e_low

	data_scaled0 = data_scaled[labels == 0]

	pos_scaled0 = np.mean(data_scaled0)
	
	shift = peak0 - pos_scaled0

	data_scaled_shifted = data_scaled + shift

	return data_scaled_shifted

def entropyFromDist(data, labels=None, drawPlot=False):
	
	if labels is None:
		data_scaled = data
	else:
		peak_sep_ev = e_high - e_low		

		data0 = data[labels == 0]
		data1 = data[labels == 1]

		pos0 = np.mean(data0)
		pos1 = np.mean(data1)

		deltapeak = np.abs(pos0-pos1)

		scale = peak_sep_ev/deltapeak

		if drawPlot:
			fig = plt.figure()
			fig.suptitle("Scaled To Energy Scale")

			ax1 = fig.add_subplot(121)
			ax1.hist(data0, bins='auto', alpha=0.5)
			ax1.hist(data1, bins='auto', alpha=0.5)	

			ax2 = fig.add_subplot(122)
			ax2.hist(data*scale, bins='auto')
			
			plt.show()

		data_scaled = data*scale
		

	nValues = np.size(data_scaled)
	
	minVal = np.amin(data_scaled)-1
	maxVal = np.amax(data_scaled)+1
	
	binWidth = 10
	
	nBins = int((maxVal-minVal)//binWidth + 2)


	bins_list = np.linspace(minVal, minVal+binWidth*nBins, nBins, endpoint=False)

	histogram = np.histogram(data_scaled, bins=bins_list)
	probs = histogram[0]/nValues

	ent = -(probs*np.ma.log(probs)).sum()

	if drawPlot:
		
		#print("nValues", nValues)
		#print("minVal", minVal)
		#print("maxVal", maxVal)
		#print("nBins", nBins)
		#print("bins_list", bins_list)

		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.hist(data_scaled, bins=bins_list)
		ax.set_title("Data binned for Entropy Calculation")
		plt.show()	

	return ent

def getEntropy2D(degree, *params):

	points, guess = params

	theta = np.radians(degree[0])
	c, s = np.cos(theta), np.sin(theta)
	R = np.array(((c, s), (-s, c)))

	direction = R @ guess

	data = project2DScatter(points, direction=direction)	
	ent = entropyFromDist(data)

	return ent

def getEntropy3D(degree, *params):
	points, guess = params

	unit_guess = guess / np.linalg.norm(guess)

	theta = np.radians(degree[0])
	
	perp_guess = np.array([unit_guess[1], -unit_guess[0], 0])

	R = transform.Rotation.from_rotvec(theta * perp_guess).as_matrix()
	
	direction = R @ unit_guess

	data = project3DScatter(points, direction=direction)
	ent = entropyFromDist(data)

	return ent
	

def optimizeEntropy2D(points, direction_g=[8,5], d_range=90, interval=1):
	
	unit_direction_g = direction_g/np.linalg.norm(direction_g)

	params = (points, unit_direction_g)
	
	values = slice(-d_range, d_range, interval)
	
	opt = optimize.brute(getEntropy2D, (values,), params)
	
	theta = np.radians(opt[0])
	c, s = np.cos(theta), np.sin(theta)
	R = np.array(((c, s), (-s, c)))

	direction = R @ unit_direction_g

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(points[:,0], points[:,1], marker="x")
	ax.set_xlabel("PC #1")
	ax.set_ylabel("PC #2")
	ax.plot([0, 3*direction[0]], [0, 3*direction[1]], color='green', label='Optimized Projection')
	#ax.plot([0, 3*unit_direction_g[0]], [0, 3*unit_direction_g[1]], color='orange', label='By Eye')
	ax.set_title("2D PCA Reduction")
	ax.legend(loc='upper right')
	plt.show()

	return direction

def optimizeEntropy3D(points, direction_g=[8,5,0], d_range=90, interval=1):

	# get the optimal direction in first 2 PC dimensions
	unit_direction_2d = np.append(optimizeEntropy2D(points[:,:2], direction_g=direction_g[:2]), 0)
	print("unit_direction_2d: ", unit_direction_2d)
	print("norm: ", np.linalg.norm(unit_direction_2d))

	params = (points, unit_direction_2d)

	values = slice(-d_range, d_range, interval)

	opt = optimize.brute(getEntropy3D, (values,), params)

	perp_guess = np.array([unit_direction_2d[1], -unit_direction_2d[0], 0])
	R = transform.Rotation.from_rotvec(np.radians(opt[0]) * perp_guess).as_matrix()	

	direction = R @ unit_direction_2d

	fig = plt.figure()
	ax = plt.axes(projection='3d')
	ax.scatter(points[:,0], points[:,1], points[:,2], marker="x")
	ax.plot([0, 3*direction[0]], [0, 3*direction[1]], [0, 3*direction[2]], color='green', label='Optimized Direction')
	ax.plot([0, 3*unit_direction_2d[0]], [0, 3*unit_direction_2d[1]], [0, 3*unit_direction_2d[2]], color='orange', label='By Eye')
	ax.set_xlabel("PC #1")
	ax.set_ylabel("PC #2")
	ax.set_zlabel("PC #3")
	ax.set_title("3D PCA Reduction")
	ax.legend(loc='upper right')
	plt.show()

	return direction


def rotate3D(degree, ortho1, direction):
	unit_direction  = direction/np.linalg.norm(direction)
	ortho1 = ortho1/np.linalg.norm(ortho1)	

	theta = np.radians(degree)
	
	azim = theta[0]
	polar = theta[1]

	R1 = transform.Rotation.from_rotvec(theta[0] * ortho1).as_matrix()

	direction_r1 = R1 @ direction
	unit_direction_r1 = direction_r1/np.linalg.norm(direction_r1)

	R2 = transform.Rotation.from_rotvec(theta[1] * unit_direction).as_matrix()

	direction_r2 = R2 @ direction_r1

	return direction_r2

def getEntropy3D_1step(degree, *params):
	points, ortho1, unit_direction_g = params

	unit_direction = rotate3D(degree, ortho1, unit_direction_g)
	unit_direction = unit_direction/np.linalg.norm(unit_direction)

	data = project3DScatter(points, direction=unit_direction)

	ent = entropyFromDist(data)

	return ent
	

def optimizeEntropy3D_1step(points, direction_g=[8,5,0], d_range=90, interval=1):
	
	direction_g = np.array(direction_g)

	unit_direction_g = direction_g/np.linalg.norm(direction_g)
	
	x = np.random.randn(3)
	x = x - x.dot(unit_direction_g) * unit_direction_g

	ortho1 = x/np.linalg.norm(x)

	params = (points, ortho1, unit_direction_g)

	azim = slice(-d_range, d_range, interval)
	polar = slice(0, 360, 1)
	
	opt = optimize.brute(getEntropy3D_1step, (azim, polar), params)

	direction = rotate3D(opt, ortho1, direction_g)
	unit_direction = direction/np.linalg.norm(direction)

	fig = plt.figure()
	ax = plt.axes(projection='3d')
	ax.scatter(*np.rollaxis(points, 1), marker='x')
	
	guess_points = np.array([[0,0,0], 3*unit_direction_g]).T
	opt_points = np.array([[0,0,0], 3*unit_direction]).T

	ax.plot(*opt_points, color='green', label='Optimized Direction')
	#ax.plot(*guess_points, color='orange', label='Guess')

	ax.set_xlabel("PC #1")
	ax.set_ylabel("PC #2")
	ax.set_zlabel("PC #3")
	
	ax.set_title("Fe55 3D PCA Reduction")

	ax.legend(loc='upper right')

	plt.show()

	return unit_direction

def showSearchPoints3D(direction=[8,5,0]):
	unit_direction = direction/np.linalg.norm(direction)

	azim_n = 10
	polar_n = 40
	N = azim_n * polar_n

	vects = np.zeros(shape=(N, 3))
	
	azim = np.linspace(0, 45, azim_n)
	polar = np.linspace(0, 360, polar_n)	

	x = np.random.randn(3)
	x = x - x.dot(unit_direction) * unit_direction
	ortho1 = x/np.linalg.norm(x)

	for i, a in enumerate(azim):
		for j, p in enumerate(polar):
			n = i*polar_n + j
			vect = rotate3D([a,p], ortho1, direction)
			vects[n] = vect
	
	print(vects.shape)
	print(vects)

	fig = plt.figure()
	ax = plt.axes(projection='3d')
	
	direction_points = np.array([[0,0,0], direction]).T
	
	ax.plot(*direction_points, color='green')
	ax.scatter(*np.rollaxis(vects, 1), marker='x')
	
	plt.show()

def allVectsND(dim, norm, steps=10):
	
	nvects = (2*steps)**(dim-1)
	split = nvects//2
	vlist = np.zeros(shape=(nvects, dim))
	
	if dim==1:
		vlist[0] = norm
	else:
		group = (2*steps)**(dim-2)
		
		if (dim-1)==1:
			start = 0
		else:
			start = norm/steps
		for i, n in enumerate(np.linspace(start, norm, steps)):
			vlist_short = allVectsND(dim-1, n, steps)
			vlist[i*group:(i+1)*group, 1:] = vlist_short
			vlist[split+i*group:split+(i+1)*group, 1:] = vlist_short
			vlist[i*group:(i+1)*group, 0] = np.sqrt(norm**2-n**2)
			vlist[split+i*group:split+(i+1)*group, 0] = -np.sqrt(norm**2-n**2)
	
	return vlist

def plotEntropy(dim, samples=1000, showProjection=False):

	'''
	Plot the entropy landscape in 2 or 3 dimensions
	'''
	
	if not (dim == 2 or dim == 3 or dim == 4):
		raise ValueError("Can only plot for 2 or 3 or 4 dimensions.")

	points, labels = generateScatter_labeled(dim, mkid.loadTraces())
	phi = np.linspace(0, 180, samples)
	theta = np.linspace(0, 180, samples)

	if dim == 2:
		ent = np.zeros(samples)
		for i, p in enumerate(phi):
			ent[i] = entropyFromSpherical([p], points, labels, 1, False)

		fig = plt.figure()
		ax = plt.axes()
		ax.plot(phi, ent)
		
		ax.set_xlabel("Phi [degrees]")
		ax.set_ylabel("Shannon Entropy")
		ax.set_title("Entropy in 2D")

		plt.show()

	elif dim == 3:
		ent = np.zeros(shape=(samples, samples))
		for i, p in enumerate(phi):
			for j, t in enumerate(theta):
				ent[i, j] = entropyFromSpherical([p,t], points, labels, 1, False)
		
		P, T = np.meshgrid(phi, theta)
	
		fig = plt.figure()
		ax = plt.axes(projection='3d')
		ax.plot_surface(P, T, ent)
		
		ax.set_xlabel("Phi [degrees]")
		ax.set_ylabel("Theta [degrees]")
		ax.set_zlabel("Shannon Entropy")

		ax.set_title("Entropy in 3D")

		plt.show()
	
	elif dim == 4:
		ent = np.zeros(shape=(samples, samples, samples))
		for i, p in enumerate(phi):
			for j, t1 in enumerate(theta):
				for k, t2 in enumerate(theta):
					ent[i, j, k] = entropyFromSpherical([p,t1,t2], points, labels, 1, False)
	
	print("Minimum entropy: ", np.amin(ent))
	

	direction_index = np.unravel_index(np.argmin(ent, axis=None), ent.shape)
	direction = np.zeros(len(direction_index))
	for i, loc in enumerate(direction_index):
		direction[i] =  theta[loc]
	direction_c = nSphereToCartesian(direction)
	data = projectScatter(direction_c, points)
	energies = hist.distToEV(data)
	
	fwhm_list = hist.getFWHM_separatePeaks(energies, npeaks=2, bw_list=[.15,.2], desc=(str(dim) + "D PCA with Optimized Projection"), xlabel="Energy [eV]", drawPlot=showProjection)

	print("Best direction (spherical):", direction)
	print("Best direction (cartesian):", direction_c)

def optimizeEntropyNSphere(dim=3, comp_list=None, start_coords=[], traces=None, points=None, labels=None, interval=1, npeaks=2, bw_list=[.15,.2], seed=1234, drawPlot=False, verbose=True):
	
	start_coords = np.array(start_coords)	

	if start_coords is not []:
		if len(start_coords) > dim-2:
			raise ValueError("Number of start start_coords must be less than dimension - 1")

	if comp_list is None:
		comp_list = list(range(1, dim+1))
		usingCustomComps = False
	else:
		usingCustomComps = True	

	dim = len(comp_list)
	
	if (points is None) or (labels is None):
		print("No points given")
		print("Extracting traces from file...")
		if traces is None:
			traces = mkid.loadTraces()

		print("Getting PCA decomposition in " + str(dim) + " dimensions...")
		
		points, labels = generateScatter_labeled_nthComps(comp_list=comp_list, traces=traces)

	norm = 1

	params = (points, labels, norm, False)

	# create list of bounds that has the length of our optimization input (opt_dim)
	# if no start coordinates are provided, this will have length dim
	bounds = []
	opt_dim = dim-1-len(start_coords)	
	for i in range(opt_dim):
		bounds.append((0,180))
	
	# set optimization parameters
	popsize = 350
	tol=0.0001
	mutation=1
	maxiter = dim*1000

	# create wrapper for optimization function
	# allows us to use start coordinates if supplied
	func = lambda x, *params : entropyFromSpherical([*start_coords, *x], *params)

	# optimize
	opt = optimize.differential_evolution(func, bounds, args=params, maxiter=maxiter, popsize=popsize, tol=tol, mutation=mutation, seed=seed)
	
	# complete the final coordinate set if we used start coords
	opt.x = np.array([*start_coords, *opt.x])	
	if verbose:
		print(opt)

	ent_min = opt.fun
	if verbose:
		print("Minimum entropy found: ", ent_min)

	if not opt.success:
		print(opt.message)

	# compute energies from projections
	direction = nSphereToCartesian(*opt.x)
	data = projectScatter(direction, points)
	energies = hist.distToEV(data)

	# plot intermediate data for debugging	
	if drawPlot:
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.hist(data, bins='auto')
		ax.set_title("Raw Data - Not Energies")
		

		fig2 = plt.figure()
		ax2 = fig2.add_subplot(111)
		ax2.hist(energies, bins='auto')
		ax2.set_title("Energy Data")

		plt.show()
		
	# calculate fwhm of peaks
	fwhm_list = hist.getFWHM_separatePeaks(energies, npeaks=npeaks, bw_list=bw_list, desc=(str(dim) + "D PCA Optimization Using Components " + str(comp_list)), xlabel="Energy [eV]", drawPlot=drawPlot)

	# save resutls to file if using first n components (not custom comp list)
	if not usingCustomComps:
		# create file for saving optimization results
		key = "dim" + str(dim)
		others_key = "others"
		if not os.path.isfile(db_path):
			pickle.dump({}, open(db_path, "wb"))

		db = pickle.load(open(db_path, "rb"))
	
		# create list for adding optimization results to	
		if (others_key not in db):
			db[others_key] = []

		# creat dictionary for every optimization
		entry = {"dimension": dim, "entropy": ent_min, "spherical": opt.x.tolist(), "spherical_start": start_coords, "fwhm": fwhm_list.tolist(), "popsize": popsize, "tol": tol, "mutation": mutation, "seed": seed, "nfev": opt.nfev, "nit": opt.nit}		

		# add key to db if it is result with minimum entropy
		# otherwise add to list of non-minimums
		if (key not in db) or ((key in db) and (db[key]["entropy"] > ent_min)):
			if key in db:
				db[others_key].append(db[key])
			db[key] = entry	
		else:
			db[others_key].append(entry)	
	
		# save db to file
		pickle.dump(db, open(db_path, "wb"))



	# convert coordinates to cartesian
	if verbose:	
		cart = nSphereToCartesian(opt.x[0], *opt.x[1:], norm=1)
		print("Cartesian: ", cart)
		print("x/y direction: ", cart[0]/cart[1])

	return opt, fwhm_list

def plotEnergyTimeTraces(n=3, dim=10, npeaks=2, bw_list=[.15,.2], seed=1234):
	
	traces = mkid.loadTraces()

	opt, _, comp_list = optimizeEntropyNSphere_bestComps(n=n, dim=dim, npeaks=npeaks, bw_list=bw_list, seed=seed, traces=traces, drawPlot=False)

	direction = nSphereToCartesian(*opt.x)

	points, labels = generateScatter_labeled_nthComps(comp_list=comp_list, traces=traces)

	data = projectScatter(direction, points)

	energies = distToEV_withLabels(data, labels)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(energies, marker='x', linestyle='')
	ax.grid()
	ax.set_title("Energies Over Time")
	ax.set_ylabel("Energy [eV]")
	ax.set_xlabel("Time")
	plt.show()

def optimizeEntropyCartesian_splitTraces(dim=4, s=0.5, points=None, labels=None, points1=None, points2=None, labels1=None, labels2=None, npeaks=2, bw_list=[.15,.2], seed=1234, drawPlot=True, verbose=False):

	if (points1 is None) or (points2 is None) or (labels1 is None) or (labels2 is None) or (points is None) or (labels is None):
		traces1, traces2 = mkid.loadTraces_split(s=s, seed=seed)
		points1, labels1 = generateScatter_labeled(dim=dim, traces=traces1)
		points2, labels2 = generateScatter_labeled(dim=dim, traces=traces2)
		
		traces = mkid.loadTraces()
		points, labels = generateScatter_labeled(dim=dim, traces=traces)

	if dim == 2:
		ent1_native, ent2_native, ent1_combined, ent2_combined, ent1, ent2 = optimizeEntropyNSphere_splitTraces(n=2, dim=2, s=s, bw_list=bw_list, seed=seed, drawPlot=False)
		opt1, _ = optimizeEntropyNSphere(dim=2, points=points1[:,:2], labels=labels1, seed=seed, drawPlot=False, verbose=verbose)
		opt2, _ = optimizeEntropyNSphere(dim=2, points=points2[:,:2], labels=labels2, seed=seed, drawPlot=False, verbose=verbose)
		opt, _ = optimizeEntropyNSphere(dim=2, points=points[:,:2], labels=labels, seed=seed, drawPlot=False, verbose=verbose)
		
		direction1_native = nSphereToCartesian(*opt1.x)
		direction1_native = direction1_native/np.linalg.norm(direction1_native)
	
		direction2_native = nSphereToCartesian(*opt2.x)
		direction2_native = direction2_native/np.linalg.norm(direction2_native)

		direction_combined = nSphereToCartesian(*opt.x)
		direction_combined = direction_combined/np.linalg.norm(direction_combined)	

	else:
		results = optimizeEntropyCartesian_splitTraces(dim=dim-1, points=points, labels=labels, points1=points1, points2=points2, labels1=labels1, labels2=labels2, npeaks=npeaks, bw_list=bw_list, seed=seed, drawPlot=False)

		start_coords1 = results[-1]["direction1_native"]
		start_coords2 = results[-1]["direction2_native"]
		start_coords = results[-1]["direction_combined"]

		args1 = (points1[:,:dim], labels1, False)
		args2 = (points2[:,:dim], labels2, False)
		args = (points[:,:dim], labels, False)	
	
		func1 = lambda x, *params : entropyFromCartesian([*start_coords1, *x], *params)
		func2 = lambda x, *params : entropyFromCartesian([*start_coords2, *x], *params)
		func = lambda x, *params : entropyFromCartesian([*start_coords, *x], *params)		

		bounds = [(-50, 50)]
		popsize = 100
		tol = 0.0001

		# generate starting population randomly
		# make sure to include 0 in the starting population so that entropy can't go down
		init = np.append(np.random.uniform(low=-50, high=50, size=(popsize-1,1)), [[0]], axis=0)

		opt1 = optimize.differential_evolution(func1, bounds, args=args1, popsize=popsize, tol=tol, init=init, seed=seed)
		opt2 = optimize.differential_evolution(func2, bounds, args=args2, popsize=popsize, tol=tol, init=init, seed=seed)
		opt = optimize.differential_evolution(func, bounds, args=args, popsize=popsize, tol=tol, init=init, seed=seed)		

		# get oprimized directions native to each set of traces	
		direction1_native = np.array([*start_coords1, *opt1.x])
		direction2_native = np.array([*start_coords2, *opt2.x])
		direction_combined = np.array([*start_coords, *opt.x])

		direction1_native = direction1_native/np.linalg.norm(direction1_native)
		direction2_native = direction2_native/np.linalg.norm(direction2_native)
		direction_combined = direction_combined/np.linalg.norm(direction_combined)

		# get the entropy of each set projected into its own optimized direction
		ent1_native = opt1.fun	
		ent2_native = opt2.fun

		# get the entropy of each set projected into the optimized direction of the other set
		ent1 = entropyFromCartesian(direction2_native, points1[:,:dim], labels1, False)
		ent2 = entropyFromCartesian(direction1_native, points2[:,:dim], labels2, False)

		ent1_combined = entropyFromCartesian(direction_combined, points1[:,:dim], labels1, False)
		ent2_combined = entropyFromCartesian(direction_combined, points2[:,:dim], labels2, False)

	result = {"dim": dim, "ent1_native": ent1_native, "ent2_native": ent2_native, "ent1": ent1, "ent2": ent2, "ent1_combined": ent1_combined, "ent2_combined": ent2_combined, "direction1_native": direction1_native, "direction2_native": direction2_native, "direction_combined": direction_combined}

	print("Direction 1: ", result["direction1_native"])
	print("Direction 2: ", result["direction2_native"])
	print("Combined Direction: ", result["direction_combined"])
	print("Entropy 1: ", result["ent1_native"])
	print("Entropy 2: ", result["ent2_native"])
	print("Entropy 1 using Direction 2: ", result["ent1"])
	print("Entropy 2 using Direction 1: ", result["ent2"])
	print("Entropy 1 using Combined Opt: ", result["ent1_combined"])
	print("Entropy 2 using Combined Opt: ", result["ent2_combined"])

	# generate list of results
	if dim == 2:
		results = [result]
	else:
		results.append(result)

	return results

def optimizeEntropyNSphere_splitTraces(n=3, dim=10, s=0.5, npeaks=2, bw_list=[.15,.2], seed=1234, drawPlot=True):
	
	# get both sets of traces
	traces = mkid.loadTraces()
	traces1, traces2 = mkid.loadTraces_split(s=s, seed=seed)

	# get optimizatiions for both sets of traces
	opt1, _, comp_list1 = optimizeEntropyNSphere_bestComps(n=n, dim=dim, npeaks=npeaks, bw_list=bw_list, seed=seed, traces=traces1, drawPlot=False, verbose=False)
	opt2, _, comp_list2 = optimizeEntropyNSphere_bestComps(n=n, dim=dim, npeaks=npeaks, bw_list=bw_list, seed=seed, traces=traces2, drawPlot=False, verbose=False)
	opt, _, comp_list = optimizeEntropyNSphere_bestComps(n=n, dim=dim, npeaks=npeaks, bw_list=bw_list, seed=seed, traces=traces, drawPlot=False, verbose=False)	

	# get oprimized directions native to each set of traces	
	direction1_native = nSphereToCartesian(*opt1.x)
	direction2_native = nSphereToCartesian(*opt2.x)
	direction_combined = nSphereToCartesian(*opt.x)

	# decompose traces into their native component list
	points1_native, labels1_native = generateScatter_labeled_nthComps(comp_list=comp_list1, traces=traces1)
	points2_native, labels2_native = generateScatter_labeled_nthComps(comp_list=comp_list2, traces=traces2)
	points, labels = generateScatter_labeled_nthComps(comp_list=comp_list, traces=traces)

	# decompose traces into the opposiite component lists
	points1, labels1 = generateScatter_labeled_nthComps(comp_list=comp_list2, traces=traces1)
	points2, labels2 = generateScatter_labeled_nthComps(comp_list=comp_list1, traces=traces2)
	points, labels = generateScatter_labeled_nthComps(comp_list=comp_list, traces=traces)

	# get the entropy of each set projected into its own optimized direction
	ent1_native = opt1.fun	
	ent2_native = opt2.fun

	# get the entropy of each set projected into the optimized direction of the other set
	ent1 = entropyFromSpherical(opt2.x, points1, labels1, 1, False)
	ent2 = entropyFromSpherical(opt1.x, points2, labels2, 1, False)

	ent1_combined = entropyFromSpherical(opt.x, points1, labels1, 1, False)
	ent2_combined = entropyFromSpherical(opt.x, points2, labels2, 1, False)

	# 1d data in native direction
	data1_native = projectScatter(direction1_native, points1_native)
	data2_native = projectScatter(direction2_native, points2_native)
	
	# 1d data in direction of other data
	data1 = projectScatter(direction2_native, points1)
	data2 = projectScatter(direction1_native, points2)

	# 1d data converted to energies
	energies1_native = distToEV_withLabels(data1_native, labels1_native)
	energies2_native = distToEV_withLabels(data2_native, labels2_native)
	energies1 = distToEV_withLabels(data1, labels1)
	energies2 = distToEV_withLabels(data2, labels2)

	# get FWHM from energy data
	fwhm_list1_native = hist.getFWHM_separatePeaks(energies1_native, npeaks=npeaks, bw_list=bw_list, desc=("Energy 1 Native " + "Comps " + str(comp_list1) + " Entropy " + str(ent1_native)), xlabel="Energy [eV]", drawPlot=drawPlot)
	fwhm_list2_native = hist.getFWHM_separatePeaks(energies2_native, npeaks=npeaks, bw_list=bw_list, desc=("Energy 2 Native " + "Comps " + str(comp_list2) + " Entropy " + str(ent2_native)), xlabel="Energy [eV]", drawPlot=drawPlot)
	fwhm_list1 = hist.getFWHM_separatePeaks(energies1, npeaks=npeaks, bw_list=bw_list, desc=("Energy 1 " + "Comps " + str(comp_list2) + " Entropy " + str(ent1)), xlabel="Energy [eV]", drawPlot=drawPlot)
	fwhm_list2 = hist.getFWHM_separatePeaks(energies2, npeaks=npeaks, bw_list=bw_list, desc=("Energy 2 " + "Comps " + str(comp_list1) + " Entropy " + str(ent2)), xlabel="Energy [eV]", drawPlot=drawPlot)

	print("Direction 1: ", direction1_native)
	print("Direction 2: ", direction2_native)
	print("Entropy 1: ", ent1_native)
	print("Entropy 2: ", ent2_native)
	print("Entropy 1 using Direction 2: ", ent1)
	print("Entropy 2 using Direction 1: ", ent2)

	return ent1_native, ent2_native, ent1_combined, ent2_combined, ent1, ent2

def plotCrossValidationCartesian(dim=20, s=0.5, npeaks=2, bw_list=[.15,.2], seed=1234, drawPlot=True):
	
	traces1, traces2 = mkid.loadTraces_split(s=s, seed=seed)
	points1, labels1 = generateScatter_labeled(dim=dim, traces=traces1)
	points2, labels2 = generateScatter_labeled(dim=dim, traces=traces2)

	traces = mkid.loadTraces()
	points, labels = generateScatter_labeled(dim=dim, traces=traces)

	results = optimizeEntropyCartesian_splitTraces(dim=dim, s=s, points=points, labels=labels, points1=points1, points2=points2, labels1=labels1, labels2=labels2, npeaks=npeaks, bw_list=bw_list, seed=seed, drawPlot=False, verbose=False)

	ent1_native_list = []
	ent2_native_list = []
	ent1_list = []
	ent2_list = []
	ent1_combined_list = []
	ent2_combined_list = []
	dim_list = []

	for result in results:
		dim_list.append(result["dim"])
		ent1_native_list.append(result["ent1_native"])
		ent2_native_list.append(result["ent2_native"])
		ent1_list.append(result["ent1"])
		ent2_list.append(result["ent2"])
		ent1_combined_list.append(result["ent1_combined"])
		ent2_combined_list.append(result["ent2_combined"])

	ent1_native_array = np.array(ent1_native_list)
	ent2_native_array = np.array(ent2_native_list)
	ent1_array = np.array(ent1_list)
	ent2_array = np.array(ent2_list)
	dim_array = np.array(dim_list)
	ent1_combined_array = np.array(ent1_combined_list)
	ent2_combined_array = np.array(ent2_combined_list)

	_, energy1_array = np.unique(labels1, return_counts=True)
	_, energy2_array = np.unique(labels2, return_counts=True)

	fig = plt.figure()
	ax1 = fig.add_subplot(121)
	ax1.plot(dim_array, ent1_native_array, label="Native")
	ax1.plot(dim_array, ent1_array, label="Non-Native")
	ax1.plot(dim_array, ent1_combined_array, label="Combined")
	ax1.set_title("Set #1 Entropies")
	ax1.set_xlabel("Dimension")
	ax1.set_ylabel("Entropy")
	ax1.legend(loc='upper right')
	ax2 = fig.add_subplot(122)
	ax2.plot(dim_array, ent2_native_array, label="Native")
	ax2.plot(dim_array, ent2_array, label="Non-Native")
	ax2.plot(dim_array, ent2_combined_array, label="Combined")
	ax2.set_title("Set #2 Entropies")
	ax2.set_xlabel("Dimension")
	ax2.set_ylabel("Entropy")
	ax2.legend(loc='upper right')

	#energy_labels = ["Low", "High"]
	#ax3 = fig.add_subplot(223)
	#ax3.pie(energy1_array, labels=energy_labels)
	#ax4 = fig.add_subplot(224)	
	#ax4.pie(energy2_array, labels=energy_labels)	

	fig2 = plt.figure()
	ax = fig2.add_subplot(111)
	ax.plot(dim_array, (ent1_array-ent1_native_array)/ent1_native_array, label="Half #1")
	ax.plot(dim_array, (ent2_array-ent2_native_array)/ent2_native_array, label="Half #2")
	ax.set_title("Cross-Validation Entropies")
	ax.set_xlabel("Dimension")
	ax.set_ylabel("Cross-Validation Entropy [Relative]")
	ax.legend(loc='upper right')
	plt.show()

def plotCrossValidation(n=4, dim=20, s=0.5, npeaks=2, bw_list=[.15,.2], seed=1234, drawPlot=True):
	
	ent1_native_list = []
	ent2_native_list = []
	ent1_combined_list = []
	ent2_combined_list = []
	ent1_list = []
	ent2_list = []
	dim_list = []

	for i in range(2, n+1):
		ent1_native, ent2_native, ent1_combined, ent2_combined, ent1, ent2 = optimizeEntropyNSphere_splitTraces(n=i, dim=dim, s=s, npeaks=2, bw_list=[.15,.2], seed=seed, drawPlot=False)
		
		dim_list.append(i)
		ent1_native_list.append(ent1_native)
		ent2_native_list.append(ent2_native)
		ent1_combined_list.append(ent1_combined)
		ent2_combined_list.append(ent2_combined)
		ent1_list.append(ent1)
		ent2_list.append(ent2)

	ent1_native_array = np.array(ent1_native_list)
	ent2_native_array = np.array(ent2_native_list)
	ent1_combined_array = np.array(ent1_combined_list)
	ent2_combined_array = np.array(ent2_combined_list)
	ent1_array = np.array(ent1_list)
	ent2_array = np.array(ent2_list)
	dim_array = np.array(dim_list)

	fig = plt.figure()
	ax1 = fig.add_subplot(121)
	ax1.plot(dim_array, ent1_native_array, label="Native")
	ax1.plot(dim_array, ent1_array, label="Non-Native")
	ax1.plot(dim_array, ent1_combined_array, label="Combined")
	ax1.set_title("Set #1 Entropies")
	ax1.set_xlabel("Dimension")
	ax1.set_ylabel("Entropy")
	ax1.legend(loc='upper right')
	ax2 = fig.add_subplot(122)
	ax2.plot(dim_array, ent2_native_array, label="Native")
	ax2.plot(dim_array, ent2_array, label="Non-Native")
	ax2.plot(dim_array, ent2_combined_array, label="Combined")
	ax2.set_title("Set #2 Entropies")
	ax2.set_xlabel("Dimension")
	ax2.set_ylabel("Entropy")
	ax2.legend(loc='upper right')

	fig2 = plt.figure()
	ax = fig2.add_subplot(111)
	ax.plot(dim_array, (ent1_array-ent1_native_array)/ent1_native_array, label="Half #1")
	ax.plot(dim_array, (ent2_array-ent2_native_array)/ent2_native_array, label="Half #2")
	ax.set_title("Cross-Validation Entropies")
	ax.set_xlabel("Dimension")
	ax.set_ylabel("Cross-Validation Entropy [Relative]")
	ax.legend(loc='upper right')
	plt.show()


def optimizeEntropyNSphere_bestComps(n=5, traces=None, dim=10, npeaks=2, bw_list=[.15,.2], drawPlot=True, verbose=True, seed=1234):
	comp_list = getImpactfulComponents_cartesian(n=n, dim=dim)

	print("Comp list: ", comp_list)
	opt, fwhm_list = optimizeEntropyNSphere(comp_list=comp_list, traces=traces, npeaks=npeaks, bw_list=bw_list, seed=seed, drawPlot=drawPlot, verbose=verbose)
	
	return opt, fwhm_list, comp_list

def optimizeEntropyNSphere_recursive(dim=7, comp_list=None, points=None, labels=None, interval=1, npeaks=2, bw_list=[.15,.2], seed=1234, drawPlot=False):
	
	if dim == 4:
		opt, fwhm_list = optimizeEntropyNSphere(dim=4, seed=seed)
	else:
		opt_lower, _ = optimizeEntropyNSphere_recursive(dim=dim-1, seed=seed)
		start_coords = opt_lower.x
		opt, fwhm_list = optimizeEntropyNSphere(dim=dim, start_coords=start_coords, seed=seed)
	
	return opt, fwhm_list

def optimizeEntropyCartesian_recursive(dim=7, points=None, labels=None, npeaks=2, bw_list=[.15,.2], seed=1234, verbose=False):

	# get points if needed
	if (points is None) or (labels is None):
		print("No labeled points given")
		print("Extracting traces from file...")
		traces = mkid.loadTraces()
		print("Getting PCA decomposition in " + str(dim) + " dimensions...")
		points, labels = generateScatter_labeled(dim=dim, traces=traces, verbose=False)
	
	# optimize in 2 dimensions as base case
	if dim == 2:
		if verbose:
			print("Optimizing in 2D...")
		opt, _ = optimizeEntropyNSphere(dim=2, points=points[:,:2], labels=labels, seed=seed, drawPlot=False, verbose=verbose)
		direction = nSphereToCartesian(*opt.x)
		direction = direction/np.linalg.norm(direction)
	else:
		# get the first n-1 components recursively
		start_coords, results = optimizeEntropyCartesian_recursive(dim=dim-1, points=points, labels=labels, seed=seed, verbose=verbose)
		
		# zip parameters
		args = (points[:,:dim], labels, False)		
		
		# mask function so we can optimize in 1D
		func = lambda x, *params : entropyFromCartesian([*start_coords, *x], *params)
		
		# arbitrary bounds for now
		bounds = [(-50, 50)]
		
		popsize=100
		tol = 0.0001	
		
		# generate starting population randomly
		# make sure to include 0 in the starting population so that entropy can't go down
		init = np.append(np.random.uniform(low=-50, high=50, size=(popsize-1,1)), [[0]], axis=0)
		
		# perform optimization		
		if verbose:
			print("Optimizing in " + str(dim) + "D...")
		opt = optimize.differential_evolution(func, bounds, args=args, popsize=popsize, tol=tol, init=init, seed=seed)

		# build direction vector and normalize
		direction = np.array([*start_coords, *opt.x])
		direction = direction/np.linalg.norm(direction)
		
	# print optimization results
	if verbose:
		print("Best direction: ", direction)
		print("Best entropy: ", opt.fun)
		if dim > 2:
			delta = opt.fun - results[-1]["entropy"]
			print("Delta entropy: ", delta)
			if delta > 0:
				print("ENTROPY INCREASED")	
		print("----------------------------------------------")

	# calculate fwhm
	data = projectScatter(direction, points[:,:dim])
	energies = hist.distToEV(data)
	fwhm_list = hist.getFWHM_separatePeaks(energies, npeaks=npeaks, bw_list=bw_list, desc=("Entropy " + str(opt.fun)), xlabel="Energy [eV]", drawPlot=False)

	# store data in dictionary for each result
	result = {"dim": dim, "entropy": opt.fun, "direction": direction, "fwhm": fwhm_list}
	
	# generate list of results
	if dim == 2:
		results = [result]
	else:
		results.append(result)

	return direction, results

def plotNDOptimization_cartesian(n=5, points=None, labels=None, seed=1234, verbose=False, drawPlot=True):
	
	'''
	Plot entropy and fwhm for the first n dimensions of principal component space. Uses recursive optimization
	in rectangular coordinates which performs a 1D minimization for each dimension. Return list of dimensions
	and list of entropies.
	'''
	
	# test up to dimension n
	if n<2:
		raise ValueError("N must be 2 or larger")
	
	dim = n

	if (points is None) or (labels is None):
		print("No labeled points given")
		print("Extracting traces from file...")
		traces = mkid.loadTraces()
		print("Getting PCA decomposition in " + str(dim) + " dimensions...")
		points, labels = generateScatter_labeled(dim=dim, traces=traces)
	
	# get optimization results
	_, results = optimizeEntropyCartesian_recursive(dim=dim, points=points, labels=labels, seed=seed, verbose=verbose)	

	# plot data
	dim_list = []
	entropy_list = []
	first_fwhm_list = []
	second_fwhm_list = []

	for obj in results:
		dim_list.append(obj["dim"])
		entropy_list.append(obj["entropy"])
		first_fwhm_list.append(obj["fwhm"][0])
		second_fwhm_list.append(obj["fwhm"][1])
	
	if drawPlot:
		fig = plt.figure()
		ax_fwhm = fig.add_subplot(121)
		ax_ent = fig.add_subplot(122)
	
		ax_fwhm.plot(dim_list, first_fwhm_list, marker='x', label="Peak #1")
		ax_fwhm.plot(dim_list, second_fwhm_list, marker='x', label="Peak #2")
		ax_fwhm.set_title("Energy Resolution")
		ax_fwhm.set_xlabel("PCA Dimension")
		ax_fwhm.set_ylabel("FWHM [eV]")
		ax_fwhm.legend(loc='upper right')

		ax_ent.plot(dim_list, entropy_list, marker='x')
		ax_ent.set_title("Minimum Entropy")
		ax_ent.set_xlabel("PCA Dimension")
		ax_ent.set_ylabel("Entropy")	

		fig.suptitle("Seed: " + str(seed))
	
		plt.show()

	return dim_list, entropy_list

def plotNDOptimization(n=5, traces=None, seed=1234, drawPlot=True):
	if n<2:
		raise ValueError("N must be 2 or larger")

	if traces is None:
		print("No traces given")
		print("Extracting traces from file...")
		traces = mkid.loadTraces()

	dim_list = []
	entropy_list = []
	first_fwhm_list = []
	second_fwhm_list = []

	for i in range(n-2):
		dim =i+3
		print("Optimizing in {}D".format(dim))
		points, labels = generateScatter_labeled(dim, traces)
		opt, fwhm = optimizeEntropyNSphere(dim=dim, points=points, labels=labels, seed=seed)
		
		dim_list.append(dim)
		entropy_list.append(opt.fun)
		first_fwhm_list.append(fwhm[0])
		second_fwhm_list.append(fwhm[1])
	
	if drawPlot:
		fig = plt.figure()
		ax_fwhm = fig.add_subplot(121)
		ax_ent = fig.add_subplot(122)
	
		ax_fwhm.plot(dim_list, first_fwhm_list, marker='x', label="Peak #1")
		ax_fwhm.plot(dim_list, second_fwhm_list, marker='x', label="Peak #2")
		ax_fwhm.set_title("Energy Resolution")
		ax_fwhm.set_xlabel("PCA Dimension")
		ax_fwhm.set_ylabel("FWHM [eV]")
		ax_fwhm.legend(loc='upper right')
	
		ax_ent.plot(dim_list, entropy_list, marker='x')
		ax_ent.set_title("Minimum Entropy")
		ax_ent.set_xlabel("PCA Dimension")
		ax_ent.set_ylabel("Entropy")	

		fig.suptitle("Seed: " + str(seed))

		plt.show()

	return dim_list, entropy_list

def plotNDOptimization_best(n=5, drawPlot=True):
	if not os.path.isfile(db_path):
		raise ValueError("Database file does not exist.")

	db = pickle.load(open(db_path, "rb"))

	keys = []
	for i in range(n):
		dim=i+1
		key = "dim" + str(dim)
		keys.append(key)
	
	dim_list = []
	entropy_list = []
	first_fwhm_list = []
	second_fwhm_list = []

	for key in keys:
		if key not in db:
			print("Database does not contain " + key + ". Skipping this key...")
		else:
			dim_list.append(db[key]["dimension"])
			entropy_list.append(db[key]["entropy"])
			first_fwhm_list.append(db[key]["fwhm"][0])
			second_fwhm_list.append(db[key]["fwhm"][1])
	
	if drawPlot:
		fig = plt.figure()
		ax_fwhm = fig.add_subplot(121)
		ax_ent = fig.add_subplot(122)
	
		ax_fwhm.plot(dim_list, first_fwhm_list, marker='x', label="Peak #1")
		ax_fwhm.plot(dim_list, second_fwhm_list, marker='x', label="Peak #2")
		ax_fwhm.set_title("Energy Resolution")
		ax_fwhm.set_xlabel("PCA Dimension")
		ax_fwhm.set_ylabel("FWHM [eV]")
		ax_fwhm.legend(loc='upper right')
	
		ax_ent.plot(dim_list, entropy_list, marker='x')
		ax_ent.set_title("Minimum Entropy")
		ax_ent.set_xlabel("PCA Dimension")
		ax_ent.set_ylabel("Entropy")	

		fig.suptitle("Best Results")

		plt.show()
	
	return dim_list, entropy_list

def plotNDOptimization_compare(n=5):
	dim_list_cart, entropy_list_cart = plotNDOptimization_cartesian(n=n, drawPlot=False)
	dim_list_best, entropy_list_best = plotNDOptimization_best(n=n, drawPlot=False)

	fig = plt.figure()
	ax_ent = fig.add_subplot(111)

	ax_ent.plot(dim_list_cart, entropy_list_cart, marker='x', label="Cartesian Results")
	ax_ent.plot(dim_list_best, entropy_list_best, marker='x', label="Best Results")
	ax_ent.set_title("Minimum Entropy")
	ax_ent.set_xlabel("PCA Dimension")
	ax_ent.set_ylabel("Entropy")

	ax_ent.legend(loc='upper right')

	plt.show()

def getImpactfulComponents_cartesian(n=5, dim=10):
	if n>dim:
		raise ValueError("Number of components requested must be lower than dimension.")

	dim_list, entropy_list = plotNDOptimization_cartesian(n=dim, drawPlot=False)

	delta_entropy_list = np.ediff1d(entropy_list)
	indices = np.argsort(delta_entropy_list)
	impact_dim_list = np.take(dim_list, indices+1)
	comp_list_long = np.insert(impact_dim_list, 0, [1,2])
	comp_list = comp_list_long[:n]

	return comp_list

def plotDeltaE(n=5, dim=10, seed=1234):
	opt, fwhm_list, comp_list = optimizeEntropyNSphere_bestComps(n=n, dim=dim, seed=seed)
	weights = nSphereToCartesian(opt.x)

	plotTrace(comp_list, weights)

def entropyFromSpherical(coords, *params):

	points, labels, norm, drawPlot = params
	
	v = nSphereToCartesian(coords[0], *coords[1:], norm=norm)
	
	data = projectScatter(v, points=points)

	ent = entropyFromDist(data, labels=labels, drawPlot=drawPlot)

	return ent

def entropyFromCartesian(v, *params):
	points, labels, drawPlot = params
	
	v = np.array(v)
	v = v/np.linalg.norm(v)

	data = projectScatter(v, points=points)
	
	ent = entropyFromDist(data, labels=labels, drawPlot=drawPlot)	

	return ent

def nSphereToCartesian(phi, *thetas, norm=1):

	thetas = np.array(thetas)
	ang = np.radians(np.insert(thetas, 0,  phi, axis=0))

	n = 1 + ang.size
	x = np.zeros(n)

	for i in range(n):
		string = ""
		x[i] = norm
		for j in range(i):
			string += (" * sin(psi" + str(j) + ") [" + str(np.sin(ang[j])) + "] = ")
			x[i] = x[i] * np.sin(ang[j])
			string += (str(x[i]))		
		if not i == n-1:
			string += (" * cos(psi" + str(i) + ") [" + str(np.cos(ang[i])) + "] = ")
			x[i] = x[i] * np.cos(ang[i])
			string += (str(x[i]))
		
		#print(string)

	return x

def span2Sphere():
	vects = np.zeros(shape=((360*180), 3))

	for i in range(360):
		for j in range(180):
			vects[180*i + j] = nSphereToCartesian(i, [j])

	return vects

def showAllVects3D(steps=10):
	#points = generate3DScatter()
	
	vects = span2Sphere()
	#vects = allVectsND(3, 1, steps=steps)
	#opt = np.array(optimizeEntropy3D_1step(points))

	fig = plt.figure()
	ax = plt.axes(projection = '3d')
	
	#direction_points = np.array([[0,0,0], opt]).T

	#ax.plot(*direction_points, color='green')
	ax.scatter(*np.rollaxis(vects, 1), marker='x')

	plt.show()

def getPCAEnergies():
	traces = mkid.loadTraces()
	points = generate2DScatter(traces)
	values = project2DScatter(points)
	energies = hist.distToEV(values)

	return energies

def optimizePCAResolution2D(points=None, npeaks=None, bw_list=None):
	
	if points is None:	
		print("No points given")
		print("Extracting traces from file...")
		traces = mkid.loadTraces()
		print("Getting PCA decomposition...")
		points = generate2DScatter(traces)

	if (bw_list is not None) and (npeaks is not None):
		if not (len(bw_list) == npeaks):
			raise ValueError("Bandwidth list must match number of peaks.")	
		
	if (npeaks is None) and (bw_list is not None):
		npeaks = len(bw_list)	

	print("Getting optimized direction...")
	direction = optimizeEntropy(points)
	print("Reducing data to 1D...")
	data = project2DScatter(points, direction=direction)
	print("Converting data to energy scale...")
	energies = hist.distToEV(data)
	print("Computing resolutions...")
	fwhm_list = hist.getFWHM_separatePeaks(energies, npeaks=npeaks, bw_list=bw_list, desc="2D PCA with Optimized Projection", xlabel="Energy [eV]", drawPlot=True)
	
	return fwhm_list

def plotProjection2D(direction=[9, 4.5], points=None, npeaks=None, bw_list=None):
	
	direction = np.array(direction)

	if points is None:	
		print("No points given")
		print("Extracting traces from file...")
		traces = mkid.loadTraces()
		print("Getting PCA decomposition...")
		points = generate2DScatter(traces)

	if (bw_list is not None) and (npeaks is not None):
		if not (len(bw_list) == npeaks):
			raise ValueError("Bandwidth list must match number of peaks.")	
		
	if (npeaks is None) and (bw_list is not None):
		npeaks = len(bw_list)	

	print("Getting direction...")
	direction = direction/np.linalg.norm(direction)
	print("Reducing data to 1D...")
	data = project2DScatter(points, direction=direction)
	print("Converting data to energy scale...")
	energies = hist.distToEV(data)
	print("Computing resolutions...")
	fwhm_list = hist.getFWHM_separatePeaks(energies, npeaks=npeaks, bw_list=bw_list, desc="2D PCA Projected onto <{0:.2f}, {1:.2f}>".format(*direction), xlabel="Energy [eV]", drawPlot=True)
	
	return fwhm_list

def optimizePCAResolution3D(points=None, npeaks=None, bw_list=None):
	if points is None:
		print("No points given")
		print("Extracting traces from file...")
		traces = mkid.loadTraces()
		print("Getting PCA decomposition...")
		points = generate3DScatter(traces)

	if (bw_list is not None) and (npeaks is not None):
		if not (len(bw_list) == npeaks):
			raise ValueError("Bandwidth list must match number of peaks.")

	if (npeaks is None) and (bw_list is not None):
		npeaks = len(bw_list)

	print("Getting optimized direction...")
	direction = optimizeEntropy3D_1step(points)
	
	print("Reducing data to 1D...")
	data = project3DScatter(points, direction=direction)
	print("Converting data to energy scale...")
	energies = hist.distToEV(data)
	print("Computing resolutions...")
	fwhm_list = hist.getFWHM_separatePeaks(energies, npeaks=npeaks, bw_list=bw_list, desc="3D PCA with Optimized Projection", xlabel="Energy [eV]", drawPlot=True)

	return fwhm_list, direction

def optimizePCAResolution(dim=3, steps=10, points=None, npeaks=None, bw_list=None):
	if points is None:
		print("No points given")
		print("Extracting traces from file...")
		traces = mkid.loadTraces()
		print("Getting PCA decomposition in " + str(dim) + " dimensions...")
		points = generateScatter(dim, traces)

	if (bw_list is not None) and (npeaks is not None):
		if not (len(bw_list) == npeaks):
			raise ValueError("Bandwidth list must match number of peaks.")

	if (npeaks is None) and (bw_list is not None):
		npeaks = len(bw_list)


	print("Getting optimized direction...")
	vects = allVectsND(dim, 1, steps=steps)
	
	data = projectScatter(vects[0], points=points)
	ent_min = entropyFromDist(data)
	direction = vects[0]
	for v in vects[1:]:
		data = projectScatter(v, points=points)
		ent = entropyFromDist(data)
		if ent< ent_min:
			ent_min = ent
			direction = v

	print("Reducing data to 1D...")
	data = projectScatter(direction, points)
	print("Converting data to energy scale...")
	energies = hist.distToEV(data)
	print("Computing resolutions...")
	fwhm_list = hist.getFWHM_separatePeaks(energies, npeaks=npeaks, bw_list=bw_list, desc=(str(dim) + "D PCA with Optimized Projection)"), xlabel="Energy [eV]", drawPlot=True)
	
	return fwhm_list, direction

def scatterAnim(angle=180, start_dir=[0,1], colors=True):
		
	start_dir = np.array(start_dir)

	points, labels = generateScatter_labeled(2)
	direction_points = np.array([[0,0], 3*start_dir]).T	

	fig = plt.figure()
	fig.set_size_inches(19.2, 10.8, True)

	ax_points = fig.add_subplot(121)
	ax_points.set_xlim(-4, 6)
	ax_points.set_ylim(-5, 4)

	ax_hist = fig.add_subplot(122)
	
	if colors:
		points0 = np.array(points[labels == 0])
		points1 = np.array(points[labels == 1])

		ax_points.scatter(points0[:,0], points0[:,1], marker='x')
		ax_points.scatter(points1[:,0], points1[:,1], marker='x')	
	else:
		ax_points.scatter(points[:,0], points[:,1], marker='x', color='b')

	draw1, = ax_points.plot(*direction_points, linewidth=3, color='g', label='')
	ax_points.set(xlabel='PC1', ylabel='PC2', title='Photon Pulses in 2D Space')

	ax_hist.hist([])
	ax_hist.set(xlabel='Projection [arb.]', ylabel='Frequency', title='1D Projection')
	ax_hist.set_xlim(-4,4)

	draw = [draw1]

	def animate(d):
		theta = np.radians(d)
	
		c, s = np.cos(theta), np.sin(theta)
		r = np.array(((c, s), (-s, c)))
		direction_r = r @ start_dir
		direction_r = np.array(direction_r)

		direction_r_points = np.array([[0,0], 3*direction_r]).T
		draw[0].set_data(*direction_r_points)
		draw[0].set_label("Angle: {0:d} degrees".format(int(d)))
		ax_points.legend(loc='upper right')
		
		dist = projectScatter(direction_r, points)		
		data_scaled = distToEV_withLabels(dist, labels)
		ent = entropyFromDist(dist, labels=labels)

		nValues = np.size(data_scaled)
	
		minVal = np.amin(data_scaled)-1
		maxVal = np.amax(data_scaled)+1
	
		binWidth = 30
	
		nBins = int((maxVal-minVal)//binWidth + 2)

		bins_list = np.linspace(minVal, minVal+binWidth*nBins, nBins, endpoint=False)

		ax_hist.clear()
		
		if colors:
			data_scaled0 = np.array(data_scaled[labels == 0])
			data_scaled1 = np.array(data_scaled[labels == 1])
			ax_hist.hist([data_scaled0, data_scaled1], stacked=True, bins=bins_list)
		else:
			ax_hist.hist(data_scaled, bins=bins_list)
		
		ax_hist.text(0.65, 0.95, "Entropy: {0:.2f}".format(ent), transform=plt.gca().transAxes)
		ax_hist.set(xlabel='Energy [eV]', ylabel='Frequency', title='1D Projection')
		ax_hist.set_xlim(4000,8000)
		ax_hist.set_ylim(0,800)

		return draw

	dsteps = np.linspace(0, angle, 300)
	anim = animation.FuncAnimation(fig, animate, frames=dsteps, interval=100)

	
	writer = animation.PillowWriter(fps=30)
	anim.save('./proj_fastish_energyspace.gif', writer=writer, dpi=100)

	#plt.show()

#fig1 = plt.figure()
#ax1 = fig1.add_subplot(121)
#ax1.semilogy(S, '-o', color='k')
#ax2 = fig1.add_subplot(122)
#ax2.plot(np.cumsum(S)/np.sum(S), '-o', color='k')
#plt.show()

