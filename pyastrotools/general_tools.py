import numpy as np
import sys, os
from scipy.optimize import least_squares, curve_fit
from scipy.linalg import svd
import datetime
from scipy.integrate import simps
from scipy.interpolate import UnivariateSpline

import matplotlib.pyplot as plt

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
			print('Downloading {}'.format(file_name))
			savefilefromurl(url, save_file)
			check=True

	else:
		print('Downloading {}'.format(file_name))
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
	

def GeneralIntegration_Trap(f, x_min, x_max, nsteps, LonSpacing=False, LogSpacing=False):
    """
    INPUTS:
        f = Function to be integrated
        x_min = Lower Bound for variable to be integrated
        x_max = Upper Bound for variable to be integrated
        LogSpacing =  x interval in log10 space. Boolean. Default = False
        LonSpacing =  x interval in loge space. Boolean. Defaut = False
        nsteps = Number of steps to use

    https://en.wikipedia.org/wiki/Trapezoidal_rule
    
    Written for Astro 530 : Stellar Atmospheres 2019

    """
    if LogSpacing:
        logx_grid = np.linspace(np.log10(x_min), np.log10(x_max), nsteps)
        logx_interval= (np.log10(x_max) - np.log10(x_min))/nsteps

        f_x = f(10**logx_grid)*(10**logx_grid)

        result = np.log(10)*(logx_interval/2) * (f_x[0] + 2*np.sum(f_x[1:-1])+ f_x[-1])

    elif LonSpacing:
        logx_grid = np.linspace(np.log(x_min), np.log(x_max), nsteps)
        logx_interval= (np.log(x_max) - np.log(x_min))/nsteps

        f_x = f(np.e**logx_grid)*(np.e**logx_grid)

        result = (logx_interval/2) * (f_x[0] + 2*np.sum(f_x[1:-1])+ f_x[-1])

    else:
        x_grid = np.linspace(x_min, x_max, nsteps)
        x_interval = (x_max - x_min)/nsteps

        f_x = f(x_grid)

        result = (x_interval/2) * (f_x[0] + 2*np.sum(f_x[1:-1])+ f_x[-1])

    return result	
    

def NumericalIntegrate1D(xarray, Matrix, xlimits, UseSimps=False):
	if UseSimps:
		Integral = simps(Matrix, xarray)
	else:
		Integral = UnivariateSpline(xarray, Matrix).integral(xlimits[0], xlimits[1])
	return Integral    


def MakeResidualPlots(x, ydata, ymodel, Title='', Xlabel='', Ylabel='', Ymodellabels=''):
	"""
	Make residual plot.
	ydata can be a list of 1D arrays, in which case will plot each one separately. In such case, give names for each dataset in Ydatalabels
	"""

	colours = ['b', 'r', 'g', 'maroon']

	NumY = np.ndim(ymodel)

	fig, axes = plt.subplots(2, 1, sharex=True)

	if np.ndim(ymodel) == 1:
		print("Dim=1")
		ymodel = [ymodel]
		Ymodellabels = [Ymodellabels]

	for i in range(NumY):
		axes[0].plot(x, ymodel[i], color=colours[i], label=Ymodellabels[i])
		R = np.sum((ydata - ymodel[i])**2)
		axes[1].plot(x, ydata - ymodel[i], color=colours[i], label="SSE = {:.4e}".format(R) )
	axes[0].plot(x, ydata, 'k', label='Data')

	plt.xlabel(Xlabel)
	axes[0].set_ylabel(Ylabel)
	axes[1].set_ylabel('Residuals')
	axes[0].set_title(Title)
	axes[0].legend()
	axes[1].legend()
	fig.subplots_adjust(hspace=0.02)
	# plt.show(block=False)
	
	return fig


def SigmaDifferenceFunction(x, dx, y, dy):
	"""
	Calculate the sigma in f, where f = x-y, and dx and dy are the standard deviations
	Return f, df, where df is the standard deviation
	"""

	return (x-y), np.sqrt(dx**2 + dy**2)

def SigmaRatioFunction(x, dx, y, dy):
	"""
	Calculate the sigma in f, where f = x/y, and dx and dy are the standard deviations
	Return f, df, where df is the standard deviation
	"""

	f = x/y
	df = np.sqrt((((dx/x)**2) + ((dy/y)**2))*f*f)

	return f, df

def SigmaLog(x, dx):
	"Calculate  dy, where y = log(x)"
	
	return dx/x
	
def SigmaAntiLog(y, dy):
	"Calculate  dx, where y = log(x)"
	
	x = np.exp(y)
		
	return dy*x
	
def SigmaLog10(x, dx):
	"Calculate  dy, where y = log10(x)"
	
	return dx/(x*np.log(10))
	

def SigmaAntiLog10(y, dy):
	"Calculate  dy, where y = log10(x)"
	
	x = 10**y
	
	return dy * x * np.log(10)

def Chi2_to_Normal(Chi2, DoF):
	return (Chi2 - DoF)/(np.sqrt(2*DoF))
