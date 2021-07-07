import numpy as np
import sys, os
from scipy.optimize import least_squares, curve_fit
from scipy.linalg import svd
import datetime


def ispositive(i):
	if i<0:
		return format(i, '.2f')
	else:
		return '+'+format(i, '.2f')

def ScaleArray(x, Bounds=[0,1]):
	Range = Bounds[1] - Bounds[0]
	#x_ = x - (1-Bounds[0])*x.min()
	Delta = (x.max() - x.min())/Range

	x_ = x/Delta

	return (x_ - x_.min()) + Bounds[0]


def LinearFit(X, Y, Yerr=None):
	
	X = X-X[0]
	arg0 = [np.median(Y)/np.median(X), 0]

	def LinFit(args):
		m = args[0]
		c = args[1]
		
		Model = X*m + c
		Data = Y
		
		if Yerr is not None:
			return (Data-Model)/Yerr
		else:
			return Data-Model


	res = least_squares(LinFit, x0=arg0)

	ysize = len(res.fun)
	cost = 2 * res.cost  # res.cost is half sum of squares!
	popt = res.x

	# Do Moore-Penrose inverse discarding zero singular values.
	_, s, VT = svd(res.jac, full_matrices=False)
	threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
	s = s[s > threshold]
	VT = VT[:s.size]
	pcov = np.dot(VT.T / s**2, VT)

	Fit = popt
	FitError = np.diag(np.sqrt(pcov))
	
	return Fit, FitError
	

def rebin(arr, new_shape):
	"""Rebin 2D array arr to shape new_shape by averaging."""
	shape = (new_shape[0], arr.shape[0] // new_shape[0],
			 new_shape[1], arr.shape[1] // new_shape[1])
	return arr.reshape(shape).mean(-1).mean(1)


def binning(bn,arr,flag=False):
	"""
	Bin 1-d array
	bn is the bin size
	arr is the array to be binned
	flag = if True then standard deviation will be returned else only the binned array
	"""
	arr=np.array(arr)
	size=np.size(arr)
	binned_size=int(size/bn)
	new_arr=np.zeros((binned_size))
	sd=np.zeros((binned_size))
	for k in range(0,binned_size):
		tmp=np.zeros((bn))
		for l in range(0,bn):
			tmp[l]=(arr[(k*bn)+l])
		new_arr[k]+=np.sum(tmp)/bn
		sd[k]=np.std(tmp)

	if flag==True:
		return new_arr,sd
	else:
		return new_arr
		

def download_file_check_staleness(url, time_tolerance, save_file, file_name):
	"""
	Check if file exists, if it does, check for staleness.
	If file is older than time_tolerance (days), then re-download.
	INPUTS:
		url : URL to download from
		time_tolerance : In days, the tolerance on staleness of file.
		save_file : Directory + file name to save the file in.
		file_name : File name for print message.
	OUTPUTS:
		check : If True, then new file downloaded, else false
	"""


	check=False

	if os.path.exists(save_file)==True:
		last_time=datetime.datetime.fromtimestamp(os.path.getmtime(save_file))
		now=datetime.datetime.now()

	# If archive file exists and is more than n days old, download newfile
		if last_time<now-datetime.timedelta(days=time_tolerance):
			print('Downloaded {}'.format(file_name))
			savefilefromurl(url, save_file)
			check=True

	else:
		print('Downloaded {}'.format(file_name))
		savefilefromurl(url, save_file)
		check=True
	return check
	

def savefilefromurl(url,saveto):
	'''
	Download the file from the url and save it at a specified location
	url: Link of the file to be saved
	saveto:Location+filename of file

	'''
	try:
		import urllib
		urllib.request.urlretrieve( url,saveto)
		print("Saved")
	except (urllib.error.URLError,IOError):
		print("Unable to save URL")	


def compactString(in_string):
	'''
	Enter string, will remove space , '-' hyphen and lower the case.
	'''
	return in_string.replace(' ', '').replace('-', '').lower()		
