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
from scipy import interpolate

import pulseanalysis.hist as hist
import pulseanalysis.data as mkid

db_path = "./pca_data/optimization.pickle"

ptrace_length = 8000
e_peaks = mkid.loadEnergies()

def plotNComponents(n, label, traces=None):
	
	'''
	Save first n principal components as images to a directory.
	'''

	if traces is None:
		print("plotNComponents(): No traces given, getting default traces...")
		traces = mkid.loadTraces()
	
	if not isinstance(label, str):
		raise ValueError("Label must be a string")
	
	# create directory based on label
	comp_path = "./decomp/" + label + "/"
	if not os.path.exists(comp_path):
		os.makedirs(comp_path)

	# subtract average from traces
	nPoints = traces.shape[0]
	traceAvg = np.mean(traces, axis=0)
	B = traces - np.tile(traceAvg, (nPoints, 1))

	# get basis
	_, S, VT = np.linalg.svd(B, full_matrices=False)

	# create figure of example trace
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(traces[0])
	ax.set_title("Full Pulse")
	ax.axes.xaxis.set_visible(False)
	ax.axes.yaxis.set_visible(False)
	plt.savefig(comp_path + "pulse.png")
	plt.close()
	
	# create figure of average trace
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(traceAvg)
	ax.set_title("Average Pulse")
	ax.axes.xaxis.set_visible(False)
	ax.axes.yaxis.set_visible(False)
	plt.savefig(comp_path + "PC0.png")
	plt.close()
	
	# compute variances
	varfrac = 100*(S**2/np.sum(S**2))

	# create figure for each principal component
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
	
	'''
	Save all traces as images to a directory.
	'''

	if traces is None:
		print("saveAllTrace(): No traces given, getting default traces...")
		traces = mkid.loadTraces()

	# save traces to directory
	for i, trace in enumerate(traces):
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot(trace)
		ax.set_title("Trace Index: " + str(i))
		ax.axes.xaxis.set_visible(False)
		ax.axes.yaxis.set_visible(False)
		plt.savefig("./traces/trace{}.png".format(i))
		plt.close()

def plotTrace(comp_list, weight_list, basis=None, title=""):
	
	'''
	Plot sum of traces given a list of components and weights.
	'''

	comp_list = np.array(comp_list)
	weight_list = np.array(weight_list)
	weight_list = weight_list/np.linalg.norm(weight_list)
	
	if comp_list.size != weight_list.size:
		raise ValueError("Weights array must have same size as component list")
	
	if basis is None:
		print("plotTrace(): No basis given, getting default basis...")
		basis = getPCABasis()
	
	VT = basis
	
	result = np.zeros_like(VT[0,:])

	# sum components
	for comp, weight in zip(comp_list, weight_list):
		comp_trace = VT[comp-1,:]
		result += comp_trace*weight

	# plot the result
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(result)
	ax.set_xlabel("Time")
	ax.set_ylabel("Phase Shift")
	ax.set_title(title)
	#ax.axes.xaxis.set_visible(False)
	#ax.axes.yaxis.set_visible(False)
	
	plt.show()
		
def plotPrincipalVariances(traces=None):
	
	'''
	Plot the cumulative sum of variance due to adding each component.
	'''

	if traces is None:
		print("plotPrincipalVariances(): No traces given, getting default traces...")
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

	'''
	Generate a list of basis vectors using SVD given a set of data vectors.
	'''

	if traces is None:
		print("getPCABasis(): No traces given, getting default traces...")
		traces = mkid.loadTraces()

	nPoints = traces.shape[0]

	# subtract average from traces
	traceAvg = np.mean(traces, axis=0)
	B = traces - np.tile(traceAvg, (nPoints,1))

	# compute basis using svd
	_, _, VT = np.linalg.svd(B/np.sqrt(nPoints), full_matrices=0)

	return VT

def plot2DScatter(traces=None, basis=None, drawPlot=False):
	
	'''
	Given a 2D collection of data where the first dimension contains
	each vector, generate a 2-component pca and return coordinates of each
	vector in this space. Set drawPlot to False to supress plot.
	'''

	if traces is None:
		print("plot2DScatter(): No traces given, getting default traces...")
		traces = mkid.loadTraces()

	if basis is None:
		print("plot2DScatter(): No basis given, getting default basis...")
		basis = getPCABasis(traces=traces)

	# same methods as in plot3DScatter()

	nPoints = traces.shape[0]
	traceAvg = np.mean(traces, axis=0)
	B = traces - np.tile(traceAvg, (nPoints,1))

	VT = basis

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

def plot3DScatter(traces=None, basis=None, drawPlot=True):
	if traces is None:
		print("plot3DScatter(): No traces given, getting default traces...")
		traces = mkid.loadTraces()

	if basis is None:
		print("plot3DScatter(): No basis given, getting default basis...")
		basis = getPCABasis(traces=traces)

	# subtract average from traces
	nPoints = traces.shape[0]
	traceAvg = np.mean(traces, axis=0)
	B = traces - np.tile(traceAvg, (nPoints,1))

	VT = basis

	# empty array for points
	points = np.zeros(shape=(nPoints, 3))

	# create projections
	for j in range(B.shape[0]):
		x = VT[0,:] @ B[j,:].T
		y = VT[1,:] @ B[j,:].T
		z = VT[2,:] @ B[j,:].T
		points[j,:] = [x,y,z]

	# plot projections
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

def generateScatter(dim, traces=None, basis=None):
	
	if traces is None:
		print("generateScatter(): No traces given, getting default traces...")
		traces = mkid.loadTraces()

	if basis is None:
		print("generateScatter(): No basis given, getting default basis...")
		basis = getPCABasis(traces=traces)

	if not isinstance(dim, int):
		raise ValueError("Dimension must be an integer.")

	if not (dim > 0):
		raise ValueError("Dimension must be greater than zero.")	

	if basis.shape[0] < dim:
		raise ValueError("Not enough vectors in basis. Must have at least " + str(dim) + ".")

	# subtract average from traces
	nPoints = traces.shape[0]
	points = np.zeros(shape=(nPoints, dim))
	traceAvg = np.mean(traces, axis=0)
	B = traces - np.tile(traceAvg, (nPoints,1))
	
	VT = basis

	# matrix multiply to get points
	for j in range(B.shape[0]):
		points[j,:] = VT[:dim,:] @ B[j,:].T

	return points

def generateScatter_nthComps(comp_list=[1,2,3,4,9,15], traces=None, basis=None):

	# convert list to numpy array	
	comp_list = np.array(comp_list)
	dim = np.size(comp_list)

	if traces is None:
		print("generateScatter_labeled_nthComps(): No traces given, getting default traces...")
		traces = mkid.loadTraces()

	if basis is None:
		print("generateScatter_labeled_nthComps(): No basis given, getting default basis...")
		basis = getPCABasis(traces=traces)

	# make sure that our basis contains enough vectors to form scatter
	if np.amax(comp_list) > basis.shape[0]:
		raise ValueError("Not enough vectors in basis. Must have at least " + str(np.amax(comp_list)) + ".")

	# make sure traces and basis vectors have same length
	if not traces.shape[1] == basis.shape[1]:
		raise ValueError("Traces and basis vectors do not have same size.")

	# create reduced basis set with requested components
	basis_reduced = np.take(basis, (comp_list-1), axis=0)

	# generate scatter using reduced basis set
	points, labels = generateScatter_labeled(dim, traces=traces, basis=basis_reduced)

	return points, labels

def plot2DScatter_labeled(traces=None, labels=None, basis=None, peak_list=None):
	if (traces is None) or (labels is None):
		print("plot3DScatter_labeled(): No traces given, getting default traces...")
		traces, labels = mkid.loadTraces_labeled()

	if basis is None:
		print("plot3DScatter_labeled(): No basis given, getting default basis...")
		basis = getPCABasis(traces=traces)

	if peak_list is None:
		peak_list = np.arange(7)
	else:
		peak_list = np.array(peak_list)

	points = generateScatter(2, traces=traces, basis=basis)

	fig = plt.figure()
	ax = fig.add_subplot(111)

	for i in peak_list:
		curr_points = points[labels==i]
		ax.scatter(*curr_points.T, marker='x', label="Peak #{}".format(i+1))
	
	ax.legend(loc='upper right')
	
	plt.show()

	return points, labels

def projectScatter(direction, points=None, drawPlot=False):
	if points is None:
		print("projectScatter(): No points given, getting default points...")
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

def distToEV_withLabels(data, labels, e_labels=[0,1]):
	
	'''
	Scale 1-dimensional data into energy space. Data histogram must contain two peaks, and data
	points belonging to each peak must be labeled as either 0 or 1.
	'''

	label_low = e_labels[0]
	label_high = e_labels[1]

	e_low = e_peaks[label_low]
	e_high = e_peaks[label_high]

	#scale data
	peak_sep_ev = e_high - e_low		

	data0 = data[labels == label_low]
	data1 = data[labels == label_high]

	pos0 = np.median(data0)
	pos1 = np.median(data1)

	deltapeak = np.abs(pos0-pos1)

	scale = peak_sep_ev/deltapeak

	if pos0 > pos1:
		scale = -scale

	data_scaled = np.array(data*scale)

	#shift data
	peak0 = e_low

	data_scaled0 = data_scaled[labels == label_low]

	pos_scaled0 = np.mean(data_scaled0)
	
	shift = peak0 - pos_scaled0

	data_scaled_shifted = data_scaled + shift

	return data_scaled_shifted

def entropyFromDist(data, labels, e_labels=[0,1], drawPlot=False):
	
	e_high = e_peaks[e_labels[0]]
	e_low = e_peaks[e_labels[1]]

	#if labels is None:
	#	data_scaled = data
	#else:
	#	peak_sep_ev = e_high - e_low		

	#	data0 = data[labels == 0]
	#	data1 = data[labels == 1]

	#	pos0 = np.mean(data0)
	#	pos1 = np.mean(data1)

	#	deltapeak = np.abs(pos0-pos1)

	#	scale = peak_sep_ev/deltapeak

	#	if drawPlot:
	#		fig = plt.figure()
	#		fig.suptitle("Scaled To Energy Scale")

	#		ax1 = fig.add_subplot(121)
	#		ax1.hist(data0, bins='auto', alpha=0.5)
	#		ax1.hist(data1, bins='auto', alpha=0.5)	

	#		ax2 = fig.add_subplot(122)
	#		ax2.hist(data*scale, bins='auto')
			
	
	#		plt.show()

	#	data_scaled = data*scale
	

	# scale data because entropy does not make sense without constant bin size
	data_scaled = distToEV_withLabels(data, labels, e_labels=e_labels)

	nValues = np.size(data_scaled)
	
	minVal = np.amin(data_scaled)-1
	maxVal = np.amax(data_scaled)+1
	
	binWidth = abs(e_high - e_low)/60

	# the number of bins we create based on the width of the distribution to achieve the
	# desired bin width
	nBins = int((maxVal-minVal)//binWidth + 2)

	# generate list of bin edges
	bins_list = np.linspace(minVal, minVal+binWidth*nBins, nBins, endpoint=False)

	# create histogram and get probabilities
	histogram = np.histogram(data_scaled, bins=bins_list)
	probs = histogram[0]/nValues

	# calculate entropy
	ent = -(probs*np.ma.log(probs)).sum()

	if drawPlot:
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.hist(data_scaled, bins=bins_list)
		ax.set_title("Data binned for Entropy Calculation")
		plt.show()	

	return ent

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
	
	'''
	Optimize projection direction in any dimension by minimizing entropy. Coordinates of result are in spherical. Details of each optimization are pickled to a file.
	'''

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
		if traces is None:
			print("optimizeEntropyNSphere(): No traces given, getting default traces...")
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
	print("Data shape: ", data.shape)
	energies = distToEV_withLabels(data, labels)

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
		
		# creat dictionary for every optimization
		entry = {"dimension": dim, "entropy": ent_min, "spherical": opt.x.tolist(), "spherical_start": start_coords, "fwhm": fwhm_list.tolist(), "popsize": popsize, "tol": tol, "mutation": mutation, "seed": seed, "nfev": opt.nfev, "nit": opt.nit}		
		
		if not os.path.isfile(db_path):
			pickle.dump({key: entry}, open(db_path, "wb"))
		else:

			db = pickle.load(open(db_path, "rb"))
	
			# create list for adding optimization results to	
			if (others_key not in db):
				db[others_key] = []

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

def plotEnergyTimeTraces(n=3, dim=10, traces=None, npeaks=2, bw_list=[.15,.2], seed=1234):
	
	'''
	Plot the energies of each trace in order to help detect drift.
	'''

	if traces is None:
		print("plotEnergyTimeTraces(): No traces given, getting default traces...")
		traces = mkid.loadTraces()

	# get optimized delta-E direction
	opt, _, comp_list = optimizeEntropyNSphere_bestComps(n=n, dim=dim, npeaks=npeaks, bw_list=bw_list, seed=seed, traces=traces, drawPlot=False)
	direction = nSphereToCartesian(*opt.x)

	# calculate energies using optimized direction
	points, labels = generateScatter_labeled_nthComps(comp_list=comp_list, traces=traces)
	data = projectScatter(direction, points)
	energies = distToEV_withLabels(data, labels)

	# plot time trace
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(energies, marker='x', linestyle='')
	ax.grid()
	ax.set_title("Energies Over Time")
	ax.set_ylabel("Energy [eV]")
	ax.set_xlabel("Time")
	plt.show()

def plotCrossValidation_cartesian(n=None, dim=20, s=0.5, npeaks=2, bw_list=[.15,.2], seed=1234, drawPlot=True):
	
	traces1, traces2 = mkid.loadTraces_split(s=s, seed=seed)
	traces0 = mkid.loadTraces()

	#results = crossValidation_cartesian(dim=dim, s=s, points=points, labels=labels, points1=points1, points2=points2, labels1=labels1, labels2=labels2, npeaks=npeaks, bw_list=bw_list, seed=seed, drawPlot=False, verbose=False)

	basis1 = getPCABasis(traces=traces1)
	basis2 = getPCABasis(traces=traces2)
	basis0 = getPCABasis(traces=traces0)

	points11_all, labels11_all = generateScatter_labeled(dim=dim, traces=traces1, basis=basis1)
	points22_all, labels22_all = generateScatter_labeled(dim=dim, traces=traces2, basis=basis2)
	points00_all, labels00_all = generateScatter_labeled(dim=dim, traces=traces0, basis=basis0)	

	direction1, _, comp_list1 = optimizeEntropyCartesian(n=n, dim=dim, points=points11_all, labels=labels11_all, drawPlot=False)
	direction2, _, comp_list2 = optimizeEntropyCartesian(n=n, dim=dim, points=points22_all, labels=labels22_all, drawPlot=False)
	direction0, _, comp_list0 = optimizeEntropyCartesian(n=n, dim=dim, points=points00_all, labels=labels00_all, drawPlot=False)

	points11, labels11 = generateScatter_labeled_nthComps(comp_list=comp_list1, traces=traces1, basis=basis1)
	points22, labels22 = generateScatter_labeled_nthComps(comp_list=comp_list2, traces=traces2, basis=basis2)
	points12, labels12 = generateScatter_labeled_nthComps(comp_list=comp_list2, traces=traces1, basis=basis2)
	points21, labels21 = generateScatter_labeled_nthComps(comp_list=comp_list1, traces=traces2, basis=basis1)
	points10, labels10 = generateScatter_labeled_nthComps(comp_list=comp_list0, traces=traces1, basis=basis0)
	points20, labels20 = generateScatter_labeled_nthComps(comp_list=comp_list0, traces=traces2, basis=basis0)

	#direction1, results1 = optimizeEntropyCartesian_recursive(dim=dim, points=points11, labels=labels11)
	#direction2, results2 = optimizeEntropyCartesian_recursive(dim=dim, points=points22, labels=labels22)
	#direction0, results0 = optimizeEntropyCartesian_recursive(dim=dim, points=points00, labels=labels00)

	ent1_native_list = []
	ent2_native_list = []
	ent1_list = []
	ent2_list = []
	ent1_combined_list = []
	ent2_combined_list = []
	dim_list = []

	if n is None:
		top=dim
	else:
		top=n

	for i in range(2, top+1):

		#print("Direction2: ", direction2[:n])
		#print("Direction1: ", direction1[:n])
		#print("Direction0: ", direction0[:n])
		#print("Points12: ", points12[:,:n])
		#print("Points21: ", points21[:,:n])
		#print("Points10: ", points10[:,:n])
		#print("Points20: ", points20[:,:n])

		ent11 = entropyFromCartesian(direction1[:i], points11[:,:i], labels11, False)
		ent22 = entropyFromCartesian(direction2[:i], points22[:,:i], labels22, False)
		ent12 = entropyFromCartesian(direction2[:i], points12[:,:i], labels12, False)
		ent21 = entropyFromCartesian(direction1[:i], points21[:,:i], labels21, False)
		ent10 = entropyFromCartesian(direction0[:i], points10[:,:i], labels10, False)
		ent20 = entropyFromCartesian(direction0[:i], points20[:,:i], labels20, False)

		#print("1 Entropy: ", ent11)
		#print("2 Entropy: ", ent22)
		#print("1 Projected on 2: ", ent12)
		#print("2 Projected on 1: ", ent21)
		#print("1 projected on Total: ", ent10)
		#print("2 projected on Total: ", ent20)

		dim_list.append(i)
		ent1_native_list.append(ent11)
		ent2_native_list.append(ent22)
		ent1_list.append(ent12)
		ent2_list.append(ent21)
		ent1_combined_list.append(ent10)
		ent2_combined_list.append(ent20)

	ent1_native_array = np.array(ent1_native_list)
	ent2_native_array = np.array(ent2_native_list)
	ent1_array = np.array(ent1_list)
	ent2_array = np.array(ent2_list)
	dim_array = np.array(dim_list)
	ent1_combined_array = np.array(ent1_combined_list)
	ent2_combined_array = np.array(ent2_combined_list)

	_, energy1_array = np.unique(labels11, return_counts=True)
	_, energy2_array = np.unique(labels22, return_counts=True)

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

def plotCrossValidation_nSphere(n=4, dim=20, s=0.5, npeaks=2, bw_list=[.15,.2], seed=1234, drawPlot=True):

	# get both sets of traces
	traces0 = mkid.loadTraces()
	traces1, traces2 = mkid.loadTraces_split(s=s, seed=seed)	
	
	# get PCA basis for each set of traces
	basis1 = getPCABasis(traces=traces1)
	basis2 = getPCABasis(traces=traces2)
	basis0 = getPCABasis(traces=traces0)

	# get points and labels for traces in their own basis
	# project traces onto all basis vectors
	points11_all, labels11_all = generateScatter_labeled(dim=dim, traces=traces1, basis=basis1)
	points22_all, labels22_all = generateScatter_labeled(dim=dim, traces=traces2, basis=basis2)
	points00_all, labels00_all = generateScatter_labeled(dim=dim, traces=traces0, basis=basis0)

	# get list of most impactful components
	comp_list1 = getImpactfulComponents_cartesian(n=n, dim=dim, points=points11_all, labels=labels11_all)
	comp_list2 = getImpactfulComponents_cartesian(n=n, dim=dim, points=points22_all, labels=labels22_all)
	comp_list0 = getImpactfulComponents_cartesian(n=n, dim=dim, points=points00_all, labels=labels00_all)
	
	# get points and labels for traces in each others bases
	# project traces onto only select basis vectors
	points11, labels11 = generateScatter_labeled_nthComps(comp_list=comp_list1, traces=traces1, basis=basis1)
	points22, labels22 = generateScatter_labeled_nthComps(comp_list=comp_list2, traces=traces2, basis=basis2)
	points00, labels00 = generateScatter_labeled_nthComps(comp_list=comp_list0, traces=traces0, basis=basis0)
	points12, labels12 = generateScatter_labeled_nthComps(comp_list=comp_list2, traces=traces1, basis=basis2)
	points21, labels21 = generateScatter_labeled_nthComps(comp_list=comp_list1, traces=traces2, basis=basis1)
	points10, labels10 = generateScatter_labeled_nthComps(comp_list=comp_list0, traces=traces1, basis=basis0)
	points20, labels20 = generateScatter_labeled_nthComps(comp_list=comp_list0, traces=traces2, basis=basis0)
	
	# initialize data lists
	ent1_native_list = []
	ent2_native_list = []
	ent1_combined_list = []
	ent2_combined_list = []
	ent1_list = []
	ent2_list = []
	dim_list = []

	for j in range(2, n+1):
		
		# compute optimum direction using first j components from comp_list
		opt1, _ = optimizeEntropyNSphere(comp_list=comp_list1[:j], npeaks=npeaks, bw_list=bw_list, seed=seed, points=points11[:,:j], labels=labels11, drawPlot=False, verbose=False)
		opt2, _ = optimizeEntropyNSphere(comp_list=comp_list2[:j], npeaks=npeaks, bw_list=bw_list, seed=seed, points=points22[:,:j], labels=labels22, drawPlot=False, verbose=False)
		opt0, _ = optimizeEntropyNSphere(comp_list=comp_list0[:j], npeaks=npeaks, bw_list=bw_list, seed=seed, points=points00[:,:j], labels=labels00, drawPlot=False, verbose=False)
	
		# get oprimized directions native to each set of traces	
		direction1 = opt1.x
		direction2 = opt2.x
		direction0 = opt0.x

		print("Direction 2: ", direction2)
		print("Points 12 Shape: ", points12[:,:j].shape)
	
		# get the entropy of each set projected into its own optimized direction
		ent11 = opt1.fun	
		ent22 = opt2.fun

		# get the entropy of each set projected into the optimized direction of the other set
		ent12 = entropyFromSpherical(direction2, points12[:,:j], labels12, 1, False)
		ent21 = entropyFromSpherical(direction1, points21[:,:j], labels21, 1, False)
		ent10 = entropyFromSpherical(direction0, points10[:,:j], labels10, 1, False)
		ent20 = entropyFromSpherical(direction0, points20[:,:j], labels20, 1, False)

		# append data to lists
		dim_list.append(j)
		ent1_native_list.append(ent11)
		ent2_native_list.append(ent22)
		ent1_combined_list.append(ent10)
		ent2_combined_list.append(ent20)
		ent1_list.append(ent12)
		ent2_list.append(ent21)
	
	# convert data to numpy arrays
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

def optimizeEntropyNSphere_bestComps(n=5, dim=15, comp_list=None, start_coords=[], traces=None, points=None, labels=None, interval=1, npeaks=2, bw_list=[.15,.2], seed=1234, drawPlot=False, verbose=True):
	comp_list = getImpactfulComponents_cartesian(n=n, dim=dim, points=points, labels=labels, seed=seed)

	print("Comp list: ", comp_list)
	opt, fwhm_list = optimizeEntropyNSphere(comp_list=comp_list, start_coords=start_coords, traces=traces, points=np.take(points, comp_list-1, axis=1), labels=labels, interval=interval, npeaks=npeaks, bw_list=bw_list, seed=seed, drawPlot=drawPlot, verbose=verbose)
	
	return opt, fwhm_list, comp_list

def optimizeEntropyCartesian_recursive(dim=7, points=None, labels=None, npeaks=2, bw_list=[.15,.2], seed=1234, verbose=False):

	# get points if needed
	if (points is None) or (labels is None):
		print("No labeled points given")
		print("Extracting traces from file...")
		traces = mkid.loadTraces()
		print("Getting PCA decomposition in " + str(dim) + " dimensions...")
		points, labels = generateScatter_labeled(dim=dim, traces=traces)
	
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
		np.random.seed(seed)
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
	energies = distToEV_withLabels(data, labels)

	fwhm_list = hist.getFWHM_separatePeaks(energies, npeaks=npeaks, bw_list=bw_list, desc=("Entropy " + str(opt.fun)), xlabel="Energy [eV]", drawPlot=False)

	# store data in dictionary for each result
	result = {"dim": dim, "entropy": opt.fun, "direction": direction, "fwhm": fwhm_list}
	
	# generate list of results
	if dim == 2:
		results = [result]
	else:
		results.append(result)

	return direction, results

def optimizeEntropyTwoPointNSphere(title, dim=3, traces=None, points=None, labels=None, npeaks=2, bw_list=[0.15,0.15,0.15,0.15,0.15,0.15,0.15], seed=1234, verbose=False, drawPlot=True):
	if (points is None) or (labels is None):
		print("optimizeEntropyCartesian_twoPoint(): No traces given, getting default traces...")
		traces, labels = mkid.loadTraces_labeled()
		points = generateScatter(dim=dim, traces=traces)

		mask = np.in1d(labels, [0,1])

		traces_twopoint = traces[mask]
		labels_twopoint = labels[mask]
		points_twopoint = points[mask]

	opt, _ = optimizeEntropyNSphere(dim=dim, points=points_twopoint, labels=labels_twopoint, seed=seed, drawPlot=False, verbose=verbose)
	#plotNDOptimization_cartesian(n=5, points=points_twopoint, labels=labels_twopoint, seed=seed, verbose=verbose, drawPlot=drawPlot)
	direction = nSphereToCartesian(opt.x)
	data = projectScatter(direction, points=points)	
	energies = distToEV_withLabels(data, labels)

	title = title + " NSphere dim: {}".format(dim)

	plotDistribution(energies, labels, title=title, bw_list=bw_list)

def plotNDOptimizationTwoPoint(n=5, points=None, labels=None, seed=1234, verbose=False, drawPlot=True):
	if (points is None) or (labels is None):
                print("optimizeEntropyCartesian_twoPoint(): No traces given, getting default traces...")
                traces, labels = mkid.loadTraces_labeled()
                points = generateScatter(dim=n, traces=traces)

                mask = np.in1d(labels, [0,1])

                traces_twopoint = traces[mask]
                labels_twopoint = labels[mask]
                points_twopoint = points[mask]

	plotNDOptimization_cartesian(n=n, points=points_twopoint, labels=labels_twopoint, seed=seed, verbose=verbose, drawPlot=drawPlot)

def optimizeEntropyTwoPointCartesian(title, n=10, dim=15, traces=None, points=None, labels=None, npeaks=2, bw_list=[0.15,0.15,0.15,0.15,0.15,0.15,0.15], seed=1234, verbose=False, drawPlot=True):
	if (points is None) or (labels is None):
                print("optimizeEntropyCartesian_twoPoint(): No traces given, getting default traces...")
                traces, labels = mkid.loadTraces_labeled()
                points = generateScatter(dim=dim, traces=traces)

                mask = np.in1d(labels, [0,1])

                traces_twopoint = traces[mask]
                labels_twopoint = labels[mask]
                points_twopoint = points[mask]

	direction, _, best_comps = optimizeEntropyCartesian(n=n, dim=dim, points=points_twopoint, labels=labels_twopoint, seed=seed, drawPlot=False, verbose=verbose)
	data = projectScatter(direction, points=np.take(points, best_comps-1, axis=1))
	energies = distToEV_withLabels(data, labels)

	title = title + " Cartesian dim: {} of {}".format(n, dim)

	plotDistribution(energies, labels, title=title, bw_list=bw_list)

def optimizeEntropyTwoPointNSphere_bestComps(title, n=10, dim=15, traces=None, points=None, labels=None, npeaks=2, bw_list=[0.15,0.15,0.15,0.15,0.15,0.15,0.15], seed=1234, verbose=False, drawPlot=True):
	if (points is None) or (labels is None):
		print("optimizeEntropyCartesian_twoPoint(): No traces given, getting default traces...")
		traces, labels = mkid.loadTraces_labeled()
		points = generateScatter(dim=dim, traces=traces)

		mask = np.in1d(labels, [0,1])

		traces_twopoint = traces[mask]
		labels_twopoint = labels[mask]
		points_twopoint = points[mask]

	opt, _, best_comps = optimizeEntropyNSphere_bestComps(n=n, dim=dim, points=points_twopoint, labels=labels_twopoint, seed=seed, drawPlot=False, verbose=verbose)
	direction = nSphereToCartesian(opt.x)
	data = projectScatter(direction, points=np.take(points, best_comps-1, axis=1))
	energies = distToEV_withLabels(data, labels)

	title = title + " NSphere dim {} of {}: {}".format(n, dim, best_comps)

	plotDistribution(energies, labels, title=title, bw_list=bw_list)

def plotDistribution(energies, labels, title="", bw_list=7*[0.15]):

	fig = plt.figure()
	ax = fig.add_subplot(121)
	ax2 = fig.add_subplot(122)
	projections = []	

	for i in range(len(e_peaks)):
		projections.append(np.median(energies[labels==i]))
	s = interpolate.InterpolatedUnivariateSpline([0, *projections], [0, *e_peaks])


	cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

	energies_transformed = s(energies)
	bin_width = abs(e_peaks[1]-e_peaks[0])/10
	for i, e in enumerate(e_peaks):
		histdata = energies_transformed[labels==i]

		bins = np.arange(np.amin(histdata)-bin_width, np.amax(histdata)+bin_width, bin_width)

		kde_x, kde_y = hist.resolveSinglePeak(data=histdata, bw=bw_list[i])
		fwhm, fwhm_h, fwhm_l, fwhm_r = hist.fwhmFromPeak(kde_x, kde_y)	

		n, _, _ = ax.hist(histdata, color=cycle[i], alpha=0.5, bins=bins)
		area = bin_width * sum(n)
		ax.plot(kde_x, area * kde_y, color=cycle[i], lw=2, label="R: {}".format(round(e_peaks[i]/fwhm,2)))
		ax.axvline(x=np.median(histdata), color=cycle[i], lw=2)
		ax.axvline(x=e, lw=2, color="black", linestyle="dashed")
	#ax.hist(histdata, stacked=True, bins=100)
	ax.set_xlabel("Energy [eV]")
	ax.set_ylabel("Counts/Bin")
	ax.set_xlim(0, 4)	
	ax.legend(loc="upper right")

	ax2.plot(projections, e_peaks, marker='x', linestyle="")
	linex = np.linspace(0, np.amax(e_peaks), 1000)
	ax2.plot(linex, linex, lw=1, linestyle="dashed", label="Linear Model")
	ax2.plot(linex, s(linex), lw=1, label="Calibration Spline")
	ax2.set_xlabel("Projected Energy [eV]")
	ax2.set_ylabel("Real Energy [eV]")
	ax2.legend(loc="upper right")
	
	fig.suptitle(title)

	plt.show()
	
	filename = "./pca_data/" + title.replace(" ", "_")
	filename_ext = filename + "_#0.png"
	i = 1
	while os.path.exists(filename_ext):
		filename_ext = filename + "_#{}.png".format(i)
		i+=1
	plt.savefig(filename_ext)

	#plot2DScatter_labeled(traces=traces_twopoint, labels=labels_twopoint)

	#direction, fwhm_list, comp_list = optimizeEntropyCartesian(n=n, dim=dim, traces=traces_twopoint, points=points_twopoint, labels=labels_twopoint, npeaks=npeaks, bw_list=bw_list, seed=seed, verbose=verbose, drawPlot=drawPlot)

	#points_reduced = np.take(points, (comp_list-1), axis=1)
	#projectScatter(direction, points=points, drawPlot=True)

def optimizeEntropyCartesian(n=None, dim=100, traces=None, points=None, labels=None, npeaks=2, bw_list=[.15,.2], seed=1234, verbose=False, drawPlot=True):

	comp_list = np.arange(dim)+1

	if (points is None) or (labels is None):
		if traces is None:
			print("optimizeEntropyCartesian(): No traces given, getting default traces...")
			traces = mkid.loadTraces()

		print("Getting PCA decomposition in " + str(dim) + " dimensions...")
		
		points, labels = generateScatter_labeled(dim=dim, traces=traces)
		
	if n is not None:
		comp_list = getImpactfulComponents_cartesian(n=n, dim=dim, points=points, labels=labels, seed=seed)
		points = np.take(points, comp_list-1, axis=1)
		#print("Points shape: ", points.shape)
		#print("Labels size: ", labels.shape)

	dim = comp_list.size

	direction, results = optimizeEntropyCartesian_recursive(dim=dim, points=points, labels=labels, npeaks=npeaks, bw_list=bw_list, seed=seed, verbose=verbose)
	
	ent = results[-1]["entropy"]

	data = projectScatter(direction, points[:,:dim])
	energies = distToEV_withLabels(data, labels)
	fwhm_list = hist.getFWHM_separatePeaks(energies, npeaks=npeaks, bw_list=bw_list, desc=("Entropy " + str(ent)), xlabel="Energy [eV]", drawPlot=drawPlot)

	return direction, fwhm_list, comp_list

def getEnergiesCartesian(n=30, dim=80, traces=None, points=None, lables=None, npeaks=2, bw_list=[.15,.2], seed=1234, verbose=False, drawPlot=False):
	
	if (points is None) or (labels is None):
		if traces is None:
			print("getEnergiesCartesian(): No traces given, getting default traces...")
			traces = mkid.loadTraces()

		print("Getting PCA decomposition in " + str(dim) + " dimensions...")
		points, labels = generateScatter_labeled(dim=dim, traces=traces)
	
	direction, fwhm_list, comp_list = optimizeEntropyCartesian(n=n, dim=dim, traces=traces, points=points, labels=labels, npeaks=npeaks, bw_list=bw_list, seed=seed, verbose=verbose, drawPlot=drawPlot)
	
	print("Points shape: ", points.shape)
	points_reduced = np.take(points, comp_list-1, axis=1)
	print("Points reduced shape: ", points_reduced.shape)

	data = projectScatter(direction, points_reduced)
	energies = distToEV_withLabels(data, labels)

	return energies
	

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

		fig.suptitle("PCA with Phase and Dissipation. Seed: " + str(seed))
	
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

def getImpactfulComponents_cartesian(n=5, dim=10, points=None, labels=None, seed=1234):
	if n>dim:
		raise ValueError("Number of components requested must be lower than dimension.")

	dim_list, entropy_list = plotNDOptimization_cartesian(n=dim, points=points, labels=labels, seed=seed, drawPlot=False)

	delta_entropy_list = np.ediff1d(entropy_list)
	indices = np.argsort(delta_entropy_list)
	impact_dim_list = np.take(dim_list, indices+1)
	comp_list_long = np.insert(impact_dim_list, 0, [1,2])
	comp_list = comp_list_long[:n]

	return comp_list

def plotDeltaE(n=5, dim=10, title="", seed=1234):
	opt, _, comp_list = optimizeEntropyNSphere_bestComps(n=n, dim=dim, seed=seed)
	weight_list = nSphereToCartesian(opt.x)

	plotTrace(comp_list, weight_list, title=title)

def plotDeltaE_cartesian(n=None, dim=100, title="", seed=1234):
	weight_list, _, comp_list = optimizeEntropyCartesian(n=n, dim=dim, seed=seed)

	plotTrace(comp_list, weight_list, title=title)

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

