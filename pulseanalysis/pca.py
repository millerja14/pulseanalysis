import mkidcalculator as mc
import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import scipy.spatial.transform as transform

import pulseanalysis.hist as hist
import pulseanalysis.data as mkid

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

	if False:
		ex_trace = traces[np.argmin(points[:,2])]
	
		fig = plt.figure()
		ax1 = fig.add_subplot(121)
		ax1.plot(ex_trace)
		ax1.set_title("Outlier")
		ax2 = fig.add_subplot(122)
		ax2.plot(traces[0])
		ax2.set_title("Standard")
		plt.show()

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

def getEntropy(degree, *params):

	points, guess, bins = params

	theta = np.radians(degree[0])
	c, s = np.cos(theta), np.sin(theta)
	R = np.array(((c, s), (-s, c)))

	direction = R @ guess

	data = project2DScatter(points, direction=direction)	

	hist = np.histogram(data, bins=bins, density=True)[0]
	ent = -(hist*np.ma.log(hist)).sum()

	return ent

def getEntropy3D(degree, *params):
	points, guess, bins = params

	unit_guess = guess / np.linalg.norm(guess)

	theta = np.radians(degree[0])
	
	perp_guess = np.array([unit_guess[1], -unit_guess[0], 0])

	R = transform.Rotation.from_rotvec(theta * perp_guess).as_matrix()
	
	direction = R @ unit_guess

	data = project3DScatter(points, direction=direction)

	hist = np.histogram(data, bins=bins, density=True)[0]
	ent = -(hist*np.ma.log(hist)).sum()

	return ent
	

def optimizeEntropy(points, direction_g=[8,5], d_range=90, interval=1):
	
	unit_direction_g = direction_g/np.linalg.norm(direction_g)

	params = (points, unit_direction_g, 100)
	
	values = slice(-d_range, d_range, interval)
	
	opt = optimize.brute(getEntropy, (values,), params)
	
	theta = np.radians(opt[0])
	c, s = np.cos(theta), np.sin(theta)
	R = np.array(((c, s), (-s, c)))

	direction = R @ unit_direction_g

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(points[:,0], points[:,1], marker="x")
	ax.plot([0, 3*direction[0]], [0, 3*direction[1]], color='green', label='Optimized')
	ax.plot([0, 3*unit_direction_g[0]], [0, 3*unit_direction_g[1]], color='orange', label='By Eye')
	ax.legend(loc='upper right')
	plt.show()

	return direction

def optimizeEntropy3D(points, direction_g=[8,5,0], d_range=90, interval=1):

	# get the optimal direction in first 2 PC dimensions
	unit_direction_2d = np.append(optimizeEntropy(points[:,:2], direction_g=direction_g[:2]), 0)
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
	ax.plot([0, 3*direction[0]], [0, 3*direction[1]], [0, 3*direction[2]], color='green', label='Optimized')
	ax.plot([0, 3*unit_direction_2d[0]], [0, 3*unit_direction_2d[1]], [0, 3*unit_direction_2d[2]], color='orange', label='By Eye')
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

	hist = np.histogram(data, bins=bins, density=True)[0]
	ent = -(hist*np.ma.log(hist)).sum()

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

	ax.plot(*opt_points, color='green', label='Optimized')
	ax.plot(*guess_points, color='orange', label='Guess')

	ax.legend(loc='upper right')

	plt.show()

	return unit_direction

def showSearchPoints(direction=[8,5,0]):
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

def getPCAEnergies():
	traces = mkid.loadTraces()
	points = generate2DScatter(traces)
	values = project2DScatter(points)
	energies = hist.distToEV(values)

	return energies

def optimizePCAResolution(points=None, npeaks=None, bw_list=None):
	
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
	fwhm_list = hist.getFWHM_separatePeaks(energies, npeaks=npeaks, bw_list=bw_list, desc="PCA Optimized Projection 2D", xlabel="Energy [eV]", drawPlot=True)
	
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
	fwhm_list = hist.getFWHM_separatePeaks(energies, npeaks=npeaks, bw_list=bw_list, desc="PCA Optimized Projection 3D", xlabel="Energy [eV]", drawPlot=True)

	return fwhm_list


#fig1 = plt.figure()
#ax1 = fig1.add_subplot(121)
#ax1.semilogy(S, '-o', color='k')
#ax2 = fig1.add_subplot(122)
#ax2.plot(np.cumsum(S)/np.sum(S), '-o', color='k')
#plt.show()

