import mkidcalculator as mc
import numpy as np

import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
mpl.rcParams['font.size'] = 22
mpl.rcParams['lines.linewidth'] = 3.0
mpl.rcParams['axes.labelpad'] = 6.0

import scipy.optimize as optimize
import scipy.spatial.transform as transform

import pulseanalysis.hist as hist
import pulseanalysis.data as mkid

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

def entropyFromDist(data, bins):
	
	data_scaled = data
	#data_scaled = hist.distToEV(data)

	nValues = np.size(data_scaled)

	histogram = np.histogram(data_scaled, bins=bins)[0]/nValues
	ent = -(histogram*np.ma.log(histogram)).sum()

	return ent

def getEntropy2D(degree, *params):

	points, guess, bins = params

	theta = np.radians(degree[0])
	c, s = np.cos(theta), np.sin(theta)
	R = np.array(((c, s), (-s, c)))

	direction = R @ guess

	data = project2DScatter(points, direction=direction)	
	ent = entropyFromDist(data, bins)

	return ent

def getEntropy3D(degree, *params):
	points, guess, bins = params

	unit_guess = guess / np.linalg.norm(guess)

	theta = np.radians(degree[0])
	
	perp_guess = np.array([unit_guess[1], -unit_guess[0], 0])

	R = transform.Rotation.from_rotvec(theta * perp_guess).as_matrix()
	
	direction = R @ unit_guess

	data = project3DScatter(points, direction=direction)
	ent = entropyFromDist(data, bins)

	return ent
	

def optimizeEntropy2D(points, direction_g=[8,5], d_range=90, interval=1):
	
	unit_direction_g = direction_g/np.linalg.norm(direction_g)

	params = (points, unit_direction_g, 100)
	
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

	params = (points, unit_direction_2d, 100)

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
	points, ortho1, unit_direction_g, bins = params

	unit_direction = rotate3D(degree, ortho1, unit_direction_g)
	unit_direction = unit_direction/np.linalg.norm(unit_direction)

	data = project3DScatter(points, direction=unit_direction)

	ent = entropyFromDist(data, 100)

	return ent
	

def optimizeEntropy3D_1step(points, direction_g=[8,5,0], d_range=90, interval=1):
	
	direction_g = np.array(direction_g)

	unit_direction_g = direction_g/np.linalg.norm(direction_g)
	
	x = np.random.randn(3)
	x = x - x.dot(unit_direction_g) * unit_direction_g

	ortho1 = x/np.linalg.norm(x)

	params = (points, ortho1, unit_direction_g, 100)

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
			ent[i] = entropyFromSpherical([p], points, 1, 100)

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
				ent[i, j] = entropyFromSpherical([p,t], points, 1, 100)
		
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
					ent[i, j, k] = entropyFromSpherical([p,t1,t2], points, 1, 100)
	
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

def optimizeEntropyNSphere(dim, points=None, interval=1, npeaks=2, bw_list=[.15,.2]):
	
	if points is None:
		print("No points given")
		print("Extracting traces from file...")
		traces = mkid.loadTraces()
		print("Getting PCA decomposition in " + str(dim) + " dimensions...")
		points = generateScatter(dim, traces)

	norm = 1
	bins = 100

	params = (points, norm, bins)

	start = np.zeros(dim-1)
	#bounds = np.empty(dim-1, dtype='object')
	bounds = []
	for i in range(dim-1):
		#bounds[i] = (0,180)	
		bounds.append((0,180))
	print("Bounds: ", bounds)	

	#isimp = 60*np.ones(shape=(dim, dim-1))
	#for i in range(dim-1):
	#	isimp[i, i] = 120
	#isimp[-1,:] = np.ones(dim-1)*60

	#isimp = np.zeros(shape=(dim, dim-1))
	#for i in range(dim):
	#	isimp[i,:] = 180*np.random.rand(dim-1)

	#print("Initial Simplex: ", isimp)

	#opt = optimize.brute(entropyFromSpherical, (phi,), args=params)
	opt = optimize.dual_annealing(entropyFromSpherical, bounds, args=params)
	print("Minimum entropy found: ", opt.fun)


	if not opt.success:
		print(opt.message)

	direction = nSphereToCartesian(*opt.x)

	data = projectScatter(direction, points)
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.hist(data, bins='auto')
	ax.set_title("Raw Data - Not Energies")
	plt.show()
	
	energies = hist.distToEV(data)
	fwhm_list = hist.getFWHM_separatePeaks(energies, npeaks=npeaks, bw_list=bw_list, desc=(str(dim) + "D PCA with Optimized Projection"), xlabel="Energy [eV]", drawPlot=True)

	return direction, fwhm_list

def entropyFromSpherical(coords, *params):

	points, norm, bins = params
	
	v = nSphereToCartesian(coords[0], *coords[1:], norm=norm)
	
	data = projectScatter(v, points=points)

	ent = entropyFromDist(data, bins)

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
	ent_min = entropyFromDist(data, bins=100)
	direction = vects[0]
	for v in vects[1:]:
		data = projectScatter(v, points=points)
		ent = entropyFromDist(data, bins=100)
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
	

#fig1 = plt.figure()
#ax1 = fig1.add_subplot(121)
#ax1.semilogy(S, '-o', color='k')
#ax2 = fig1.add_subplot(122)
#ax2.plot(np.cumsum(S)/np.sum(S), '-o', color='k')
#plt.show()

