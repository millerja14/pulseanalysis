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

def plotNComponents(n, traces=None):
	if not isinstance(traces, np.ndarray):
		print("No traces given, getting default traces...")
		traces = mkid.loadTraces()
	
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
	plt.savefig("./decomp/pulse.png")

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(traceAvg)
	ax.set_title("Average Pulse")
	ax.axes.xaxis.set_visible(False)
	ax.axes.yaxis.set_visible(False)
	plt.savefig("./decomp/PC0.png")

	print(S)
	varfrac = 100*(S**2/np.sum(S**2))

	for i in range(n):
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot(-VT[i,:])
		ax.axes.xaxis.set_visible(False)
		ax.axes.yaxis.set_visible(False)
		ax.set_title("PC{0}: {1:.2f}% of Variance".format(i+1, varfrac[i]))
		plt.savefig("./decomp/PC{0}.png".format(i+1))

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

def entropyFromDist(data, labels=None, drawPlot=False):
	
	if labels is None:
		data_scaled = data
	else:
		peak_sep_ev = 590		

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

def plotEntropy(dim, samples=100):
	
	if not (dim == 2 or dim == 3 or dim == 4):
		raise ValueError("Can only plot for 2 or 3 or 4 dimensions.")

	points = generateScatter(dim, mkid.loadTraces())
	phi = np.linspace(0, 180, samples)
	theta = np.linspace(0, 180, samples)

	if dim == 2:
		ent = np.zeros(samples)
		for i, p in enumerate(phi):
			ent[i] = entropyFromSpherical([p], points, None, 1, False)

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
				ent[i, j] = entropyFromSpherical([p,t], points, None, 1, False)
		
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
					ent[i, j, k] = entropyFromSpherical([p,t1,t2], points, None, 1, False)
	
	print("Minimum entropy: ", np.amin(ent))
	

	direction_index = np.unravel_index(np.argmin(ent, axis=None), ent.shape)
	direction = np.zeros(len(direction_index))
	for i, loc in enumerate(direction_index):
		direction[i] =  theta[loc]
	direction_c = nSphereToCartesian(direction)
	data = projectScatter(direction_c, points)
	energies = hist.distToEV(data)
	
	fwhm_list = hist.getFWHM_separatePeaks(energies, npeaks=2, bw_list=[.15,.2], desc=(str(dim) + "D PCA with Optimized Projection"), xlabel="Energy [eV]", drawPlot=True)

	print("Best direction (spherical):", direction)
	print("Best direction (cartesian):", direction_c)

def optimizeEntropyNSphere(dim, points=None, labels=None, interval=1, npeaks=2, bw_list=[.15,.2], seed=1234, drawPlot=False):
	
	if (points is None) or (labels is None):
		print("No points given")
		print("Extracting traces from file...")
		traces = mkid.loadTraces()
		print("Getting PCA decomposition in " + str(dim) + " dimensions...")
		points, labels = generateScatter_labeled(dim, traces)

	norm = 1

	params = (points, labels, norm, False)

	start = np.zeros(dim-1)
	bounds = []
	for i in range(dim-1):
		#bounds[i] = (0,180)	
		bounds.append((0,180))

	maxiter = dim*1000
	#maxiter = 5000

	opt = optimize.differential_evolution(entropyFromSpherical, bounds, args=params, maxiter=maxiter, tol=0.0001, seed=seed)
	
	print(opt)

	#params = (points, labels, norm, False)
	#ent_min = entropyFromSpherical(opt.x, *params)
	ent_min = opt.fun
	print("Minimum entropy found: ", ent_min)

	if not opt.success:
		print(opt.message)

	direction = nSphereToCartesian(*opt.x)

	data = projectScatter(direction, points)
	
	energies = hist.distToEV(data)
	
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
		
	fwhm_list = hist.getFWHM_separatePeaks(energies, npeaks=npeaks, bw_list=bw_list, desc=(str(dim) + "D PCA. Entropy: " + str(ent_min)), xlabel="Energy [eV]", drawPlot=drawPlot)

	return opt, fwhm_list

def plotNDOptimization(n=5, traces=None, seed=1234):
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
		opt, fwhm = optimizeEntropyNSphere(dim, points=points, labels=labels, seed=seed)
		
		dim_list.append(dim)
		entropy_list.append(opt.fun)
		first_fwhm_list.append(fwhm[0])
		second_fwhm_list.append(fwhm[1])
	
	print(dim_list)
	print(entropy_list)
	print(first_fwhm_list)
	print(second_fwhm_list)

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

def entropyFromSpherical(coords, *params):

	points, labels, norm, drawPlot = params
	
	v = nSphereToCartesian(coords[0], *coords[1:], norm=norm)
	
	data = projectScatter(v, points=points)

	ent = entropyFromDist(data, labels=labels, drawPlot=drawPlot)

	return ent

def nSphereToCartesian(phi, *thetas, norm=1):
	thetas = np.array(thetas)
	ang = np.radians(np.append(thetas, phi))

	n = 1 + ang.size
	x = np.zeros(n)

	for i in range(n):
		x[i] = norm
		for j in range(i):
			x[i] = x[i] * np.sin(ang[j])	
		if i != n-1:
			x[i] = x[i] * np.cos(ang[i])
	
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

def scatterAnim(angle=180, start_dir=[0,1]):
		
	start_dir = np.array(start_dir)

	points = generateScatter(2)
	direction_points = np.array([[0,0], 3*start_dir]).T	

	fig = plt.figure()
	fig.set_size_inches(19.2, 10.8, True)

	ax_points = fig.add_subplot(121)
	ax_points.set_xlim(-4, 6)
	ax_points.set_ylim(-5, 4)

	ax_hist = fig.add_subplot(122)

	ax_points.scatter(points[:,0], points[:,1], marker='x', color='b')
	draw1, = ax_points.plot(*direction_points, linewidth=3, color='g', label='')
	ax_points.set(xlabel='PC1', ylabel='PC2', title='Photon Pulses in 2D Space')

	ax_hist.hist([], bins=100, density=True)
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
		draw[0].set_label("Angle: {0:.2f} degrees".format(d))
		ax_points.legend(loc='upper right')
		

		dist = projectScatter(direction_r, points)		
		ax_hist.clear()
		ax_hist.hist(dist, bins=50, density=True)
		ax_hist.set(xlabel='Projection [arb.]', ylabel='Frequency', title='1D Projection')
		ax_hist.set_xlim(-4,4)		

		return draw

	dsteps = np.linspace(0, angle, 100)
	anim = animation.FuncAnimation(fig, animate, frames=dsteps, interval=30)

	
	writer = animation.PillowWriter(fps=30)
	anim.save('./proj_fastish.gif', writer=writer, dpi=100)

	plt.show()

#fig1 = plt.figure()
#ax1 = fig1.add_subplot(121)
#ax1.semilogy(S, '-o', color='k')
#ax2 = fig1.add_subplot(122)
#ax2.plot(np.cumsum(S)/np.sum(S), '-o', color='k')
#plt.show()

