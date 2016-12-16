#!/usr/bin/python
#-*- coding: UTF-8 -*-

"""
Utility functions package

Author: Luciano Paz
Year:2016
"""

from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as np
from scipy import optimize
from scipy import stats
import math, os, warnings, json, re


a = np.array([ 0.886226899, -1.645349621,  0.914624893, -0.140543331])
b = np.array([-2.118377725,  1.442710462, -0.329097515,  0.012229801])
c = np.array([-1.970840454, -1.624906493,  3.429567803,  1.641345311])
d = np.array([ 3.543889200,  1.637067800])
y0 = 0.7

def erfinv(y):
	if y<-1. or y>1.:
		raise ValueError("erfinv(y) argument out of range [-1.,1]")
	if abs(y)==1.:
		# Precision limit of erf function
		x = y*5.9215871957945083
	elif y<-y0:
		z = math.sqrt(-math.log(0.5*(1.0+y)))
		x = -(((c[3]*z+c[2])*z+c[1])*z+c[0])/((d[1]*z+d[0])*z+1.0)
	else:
		if y<y0:
			z = y*y;
			x = y*(((a[3]*z+a[2])*z+a[1])*z+a[0])/((((b[3]*z+b[2])*z+b[1])*z+b[0])*z+1.0)
		else:
			z = np.sqrt(-math.log(0.5*(1.0-y)))
			x = (((c[3]*z+c[2])*z+c[1])*z+c[0])/((d[1]*z+d[0])*z+1.0)
		# Polish to full accuracy
	x-= (math.erf(x) - y) / (2.0/math.sqrt(math.pi) * math.exp(-x*x));
	x-= (math.erf(x) - y) / (2.0/math.sqrt(math.pi) * math.exp(-x*x));
	return x

_vectErf = np.vectorize(math.erf,otypes=[np.float])
_vectErfinv = np.vectorize(erfinv,otypes=[np.float])
_vectGamma = np.vectorize(math.gamma,otypes=[np.float])

def normpdf(x, mu=0., sigma=1.):
	u = (x-mu)/sigma
	return 0.3989422804014327/np.abs(sigma)*np.exp(-0.5*u*u)

def normcdf(x,mu=0.,sigma=1.):
	"""
	Compute normal cummulative distribution with mean mu and standard
	deviation sigma. x, mu and sigma can be a numpy arrays that broadcast
	together.
	
	Syntax:
	y = normcdf(x,mu=0.,sigma=1.)
	"""
	try:
		new_x = (x-mu)/sigma
	except ZeroDivisionError:
		new_x = np.sign(x-mu)*np.inf
	return 0.5 + 0.5*_vectErf(new_x / np.sqrt(2.0))

def normcdfinv(y,mu=0.,sigma=1.):
	"""
	Compute the inverse of the normal cummulative distribution with mean
	mu and standard deviation sigma. y, mu and sigma can be a numpy
	arrays that broadcast together.
	
	Syntax:
	x = normcdfinv(y,mu=0.,sigma=1.)
	"""
	x = np.sqrt(2.0)*_vectErfinv(2.*(y-0.5))
	try:
		iterator = iter(sigma)
	except TypeError:
		if sigma==0.:
			raise ValueError("Invalid sigma supplied to normcdfinv. sigma cannot be 0")
		x = sigma*x+mu
	else:
		if any(sigma==0):
			raise ValueError("Invalid sigma supplied to normcdfinv. sigma cannot be 0")
		x = sigma*x+mu
	return x

def normgamma(x,t,mu=0.,l=1.,beta=2.,alpha=2.):
	return beta**alpha/_vectGamma(alpha)*np.sqrt(0.5*l/np.pi)*t**(alpha-0.5)*np.exp(-0.5*l*t*(x-mu)**2-beta*t)

def norminvgamma(x,sigma,mu=0.,l=1.,beta=2.,alpha=2.):
	return normgamma(x,sigma**(-2),mu,l,beta,alpha)

def unique_rows(a,**kwargs):
	if a.ndim!=2:
		raise ValueError('Input array must be a two dimensional array')
	b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
	return_index = kwargs.pop('return_index', False)
	out = np.unique(b, return_index=True, **kwargs)
	idx = out[1]
	uvals = a[idx]
	if (not return_index) and (len(out) == 2):
		return uvals
	elif return_index:
		return (uvals,) + out[1:]
	else:
		return (uvals,) + out[2:]

def average_downsample(a,output_len,axis=None,ignore_nans=True,dtype=np.float):
	"""
	b = average_downsample(a,output_len,axis=None,ignore_nans=True,dtype=np.float)
	
	This function downsamples a numpy array of arbitrary shape. It does
	so by computing the average value of the input array 'a' inside a
	window of steps in order to produce an array with the supplied
	output_len. The output_len does not need to be submultiple of the
	original array's shape, and the function handles the averaging of
	the edges of each window properly.
	
	Inputs:
	- a:           np.array that will be downsampled
	- output_len:  Scalar that specifies the length of the output in the
	               downsampled axis.
	- axis:        The axis of the input array along which the array
	               will be downsampled. By default axis is None, and in
	               that case, the downsampling is performed on the
	               flattened array.
	- ignore_nans: Bool that specifies whether to ignore nans in while
	               averaging or not. Default is to ignore.
	- dtype:       Specifies the output array's dtype. Default is
	               np.float
	
	Output:
	- b:           The downsampled array. If axis is None, it will be a
	               flat array of shape equal to (int(output_len)).
	               If axis is not None, 'b' will have the same shape as
	               'a' except for the specified axis, that will have
	               int(output_len) elements
	
	Example
	>>> import numpy as np
	>>> a = a = np.reshape(np.arange(100,dtype=np.float),(10,-1))
	>>> a[-1] = np.nan
	
	>>> utils.average_downsample(a,10)
	array([  4.5,  14.5,  24.5,  34.5,  44.5,  54.5,  64.5,  74.5,  84.5,   nan])
	
	>>> utils.average_downsample(a,10,axis=0)
	array([[ 40.,  41.,  42.,  43.,  44.,  45.,  46.,  47.,  48.,  49.]])
	
	>>> utils.average_downsample(a,10,axis=1)
	array([[  4.5],
	       [ 14.5],
	       [ 24.5],
	       [ 34.5],
	       [ 44.5],
	       [ 54.5],
	       [ 64.5],
	       [ 74.5],
	       [ 84.5],
	       [  nan]])
	
	"""
	if axis is None:
		a = a.flatten()
		axis = 0
		sum_weight = 0
		window = float(a.shape[0])/float(output_len)
		b = np.zeros((int(output_len)),dtype=dtype)
	else:
		a = np.swapaxes(a,0,axis)
		sum_weight = np.zeros_like(a[0],dtype=np.float)
		b_shape = list(a.shape)
		window = float(b_shape[0]/output_len)
		b_shape[0] = int(output_len)
		b = np.zeros(tuple(b_shape),dtype=dtype)
	flat_array = a.ndim==1
	
	step_size = 1./window
	position = 0.
	i = 0
	prev_index = 0
	L = len(a)
	Lb = len(b)
	all_indeces = np.ones_like(a[0],dtype=np.bool)
	step = True
	while step:
		if ignore_nans:
			valid_indeces = np.logical_not(np.isnan(a[i]))
		else:
			valid_indeces = all_indeces
		position = (i+1)*step_size
		index = int(position)
		if prev_index==index:
			weight = valid_indeces.astype(dtype)*step_size
			sum_weight+= weight
			if flat_array:
				b[index]+= a[i]*weight if valid_indeces else 0.
			else:
				b[index][valid_indeces]+= a[i][valid_indeces]*weight[valid_indeces]
		elif prev_index!=index:
			weight = valid_indeces*(position-index)
			prev_weight = valid_indeces*(index+step_size-position)
			if flat_array:
				b[prev_index]+= a[i]*prev_weight if valid_indeces else 0.
				sum_weight+= prev_weight
				b[prev_index]/=sum_weight
				if index<Lb:
					b[index]+= a[i]*weight if valid_indeces else 0.
			else:
				b[prev_index][valid_indeces]+= a[i][valid_indeces]*prev_weight[valid_indeces]
				sum_weight+= prev_weight
				b[prev_index]/=sum_weight
				if index<Lb:
					b[index][valid_indeces]+= a[i][valid_indeces]*weight[valid_indeces]
			sum_weight = weight
		
		prev_index = index
		
		i+=1
		if i==L:
			step = False
			if index<Lb:
				b[index]/=sum_weight
	b = np.swapaxes(b,0,axis)
	return b

def holm_bonferroni(p_in,alpha=None):
	"""
	holm_bonferroni(p_in,alpha=None)
	Correct the p values for multiple tests using holm-bonferroni criteria

	Inputs:
	 p_in:  a 1D numpy array with the original p values
	 alpha: Optional input. Default is None. If not None, it must be a float
			that specifies the admisibility for the test given by the
			corrected p values.

	Outputs:
	 p2: The corrected p values
	 h:  If alpha is not None, the output is a tuple formed by (p2,h) where
		 h is an array of bools that is true or false indicating if the
		 corrected p value is lower than the supplied alpha or not
	"""
	p = p_in.flatten()
	sort_inds = np.argsort(p)
	reverse_sort_inds = np.zeros_like(sort_inds)
	for i,si in enumerate(sort_inds):
		reverse_sort_inds[si] = i
	p = p[sort_inds]
	
	p2 = np.zeros_like(p)
	m = np.sum(np.logical_not(np.isnan(p)).astype(np.int))
	for i in range(m):
		temp = (m-np.arange(i+1))*p[:i+1]
		temp[temp>1] = 1.
		p2[i] = np.max(temp)
	p2 = p2[reverse_sort_inds]
	if alpha:
		alpha = float(alpha)
		comparison = p > (alpha/(m-np.arange(m)));
		if np.all(comparison):
			h = np.zeros_like(p,dtype=np.bool)
		else:
			h = np.ones_like(p,dtype=np.bool)
			k = comparison.nonzero()[0][0]
			h[k:] = False
		return p2,h
	else:
		return p2

def corrcoef(a,b=None,axis=0,method='pearson',nanhandling='pairwise'):
	"""
	corrcoef(a,b=None,axis=0,method='pearson',nanhandling='pairwise')
	
	Input:
	 a:           Mandatory input array. Can be a 1D or 2D numpy ndarray.
	              The different dimensions of 'a' represent the observed
	              data sample, and each of the variables. This representation
	              depends on the supplied axis
	 b:           Optional input array. If 'b' is None, it is ignored. If
	              it is not None, 'b' is treated as an additional variable
	              of the array 'a'.
	 axis:        Default is 0. The 'axis' input determines which axis
	              of the supplied arrays holds different variables. If
	              axis is 0, the different variables are assumed to be
	              represented as the rows of the array, and the
	              independent samples are assumed to be in the axis=1
	              (the columns). If 'axis'=1, the array columns are
	              assumed to represent different variables and the
	              independent samples are assumed to be located in the
	              axis=0 (the rows). If axis is None, 'a' and 'b' are
	              flattened before computing the correlation coeficient
	              and each element in 'a' and 'b' are treated as
	              independent samples.
	 method:      A string indicating the method that will be used to
	              compute the correlation. Can be 'pearson', 'spearman'
	              or 'kendall'. Refer to the functions pearsonr, spearmanr
	              and kendalltau defined in scipy.stats for further
	              details
	 nanhandling: Determines how nans are handled while computing the
	              correlation coeficients. Can be None, 'pairwise' or
	              'complete'. If None, no special handling is done before
	              attempting to compute the correlation coefficients.
	              If 'complete', all the data observations that hold at least
	              one variable which is nan are discarded before computing
	              the correlation coefficients. If 'pairwise', rho[i,j] is
	              computed using rows with no NaN values in the variables
	              i or j.
	"""
	if axis is None:
		a = a.flatten()
		if not b is None:
			b = b.flatten()
		axis=1
	if a.ndim>2:
		raise ValueError("a must be a 1D or 2D numpy array")
	if not b is None:
		if b.ndim>2:
			raise ValueError("b must be None or a 1D or 2D numpy array")
		if axis==0:
			if a.ndim==1:
				a = a[None,:]
			if b.ndim==1:
				b = b[None,:]
			a = np.vstack((a,b))
		else:
			if a.ndim==1:
				a = a[:,None]
			if b.ndim==1:
				b = b[:,None]
			a = np.hstack((a,b)).T
	try:
		calculator = {'pearson':stats.pearsonr,'spearman':stats.spearmanr,'kendall':stats.kendalltau}[str(method).lower()]
	except:
		raise ValueError('Supplied method must can be "pearson", "spearman" or "kendall". Got {0} instead'.format(method))
	
	rho = np.zeros((len(a),len(a)))
	pval = np.zeros((len(a),len(a)))
	if nanhandling is None:
		for i,ai in enumerate(a):
			rho[i,i],pval[i,i] = calculator(ai,ai)
			for j,aj in enumerate(a[i+1:]):
				rho[j+i+1,i],pval[j+i+1,i] = rho[i,j+i+1],pval[i,j+i+1] = calculator(ai,aj)
	else:
		nanhandling = str(nanhandling).lower()
		if nanhandling=='pairwise':
			for i,ai in enumerate(a):
				valid = np.logical_not(np.isnan(ai))
				rho[i,i],pval[i,i] = calculator(ai[valid],ai[valid])
				for j,aj in enumerate(a[i+1:]):
					valid = np.logical_not(np.logical_or(np.isnan(ai),np.isnan(aj)))
					rho[j+i+1,i],pval[j+i+1,i] = rho[i,j+i+1],pval[i,j+i+1] = calculator(ai[valid],aj[valid])
		elif nanhandling=='complete':
			valid = np.logical_not(np.any(np.isnan(a),axis=0))
			for i,ai in enumerate(a):
				rho[i,i],pval[i,i] = calculator(ai,ai)
				for j,aj in enumerate(a[i+1:]):
					rho[j+i+1,i],pval[j+i+1,i] = rho[i,j+i+1],pval[i,j+i+1] = calculator(ai[valid],aj[valid])
		else:
			raise ValueError("nanhandling must be None, 'complete' or 'pairwise'. Got {0} instead.".format(nanhandling))
	return rho,pval

def linear_least_squares(x,y,covy=None):
	"""
	linear_least_squares(x,y,covy=None)
	
	Perform a linear least squares fit between an array x and y, and
	return the fitted parameters and the corresponding covariance
	matrix
	
	Input:
		x,y: Two 1D numpy arrays of the same shape. x is the independent
			variable and y is the dependent variable.
		covy: Can be None, a float, a 1D array or a 2D array. If None,
			the fit is performed assuming covy is the identity matrix.
			If a float, it is assumed to encode a constant standard
			deviation for every 'y' datapoint. If a 1D array, it must
			hold the standard deviations of each observed 'y'. It must
			have the same number of elements as 'y'. If it is a 2D
			array, it must have shape (len(y),len(y)) and hold the
			covariance matrix of the observations 'y'.
	
	Output:
		par: A 1D array of the fitted parameter values. The first element
			is the intercept and the second element is the slope of the
			linear fit.
		cov: A 2D array with the covariance matrix of the fitted
			parameter values. The order in which they appear is the same
			as the order of the 'par' array output.
	
	"""
	x = x.flatten(order='K')
	y = y.flatten(order='K')
	assert x.shape==y.shape, "Inputs 'x' and 'y' must have the same number of elements"
	if covy is None:
		covy = np.ones((len(y),len(y)))
	elif not isinstance(covy,np.ndarray):
		covy = np.diag(float(covy)**2*np.ones(len(y)))
	elif covy.ndim==1:
		# If covy is a vector assume it encodes the std, not the variance
		covy = np.diag(covy**2)
	elif covy.ndim!=2:
		raise ValueError("Input covy must be a None, a float, or a 1D or 2D numpy array")
	assert (len(y),len(y))==covy.shape, "Inconsistent dimensions between the supplied 'covy' input and the 'x' and 'y' arrays."
	covy = np.matrix(covy)
	mat = np.ones((len(x),2))
	mat[:,1] = x
	mat = np.matrix(mat)
	covyinv = np.linalg.inv(covy)
	cov = np.linalg.inv(mat.transpose()*covyinv*mat)
	pars = np.array(cov*mat.transpose()*covyinv*np.matrix(y.reshape((-1,1)))).flatten()
	return pars,np.array(cov)

def linear_least_squares_prediction(x,par,cov):
	"""
	linear_least_squares_prediction(x,par,cov)
	
	Get the predicted value and standard deviation for the supplied x,
	and the linear least squares fitted value and covariance matrix
	
	Input:
		x: A numpy array with the independent variable values where the
			predicted 'y' value will be computed
		par: The linear parameters as [float(intercept),float(slope)]
		cov: The covariance matrix that corresponds to the fitted
			parameters 'par'
	
	Output:
		y: The predicted value as x*slope+intercept
		sy: The standard deviation as sqrt(cov[1,1]*x**2+2*cov[0,1]*x+cov[0,0])
	
	"""
	y = par[0]+par[1]*x
	sy = np.sqrt((cov[1,1]*x+2*cov[0,1])*x+cov[0,0])
	return y,sy

def maximize_figure(fig=None):
	import matplotlib as mt
	from matplotlib import pyplot as plt
	backend = mt.get_backend().lower()
	current_figure = plt.gcf()
	if fig is None:
		fig = current_figure
	# Change the current figure to the one to be maximized
	plt.figure(fig.number)
	manager = plt.get_current_fig_manager()
	if backend in ['tkagg']:
		manager.window.state('zoomed')
	elif backend in ['wx','wxagg']:
		manager.frame.Maximize(True)
	elif backend in ['qt4agg','qt5agg']:
		manager.window.showMaximized()
	elif backend in ['gtkagg','gtk3agg','gtk','gtkcairo','gtk3cairo']:
		manager.window.gtk_window_maximize()
	elif backend=='macosx':
		plt.figure(current_figure.number)
		raise NotImplemented("Support for macosx backend is not implemented yet")
	# Revert the current figure to the one prior to the call to maximize_figure
	plt.figure(current_figure.number)

def parse_details_file():
	def clean_str(s):
		return s.partition('#')[0].strip(' \t'+os.linesep)
	def array_parser(x):
		x = x.strip(' \t\n\r')
		if x.startswith('[') and x.endswith(']'):
			return np.array([float(xx) for xx in x[1:-1].split(',')])
		else:
			raise ValueError('Supplied value {0} is not an ordinary float or list.'.format(x))
	def external_var_handler(x):
		try:
			return float(x)
		except:
			return array_parser(x)
	def time_available_to_respond_handler(x):
		try:
			return float(x)
		except ValueError:
			if x.lower()=='none':
				return float('inf')
			else:
				raise ValueError('Cannot interpret value given to time_available_to_respond')
	def extension_parser(x):
		x = str(x).strip()
		if not x.startswith('.'):
			x = '.'+x
		return x
	def data_structure_parser(x):
		try:
			return json.loads(x)
		except:
			print(x)
			raise
	options = {u'raw_data_dir':None,
				u'experiment_details':{}}
	valid_decision_model_keys = ['tp','T','iti','dt','external_var','n','reward','penalty','prior_var_prob']
	valid_fitter_keys = ['rt_cutoff','distractor','forced_non_decision_time','ISI','rt_measured_from_stim_end','time_available_to_respond','is_binary_confidence']
	valid_io_keys = ['session_parser','file_extension','time_conversion_to_seconds','excluded_files','data_structure']
	valid_experiment_details_keys = valid_decision_model_keys+valid_fitter_keys+valid_io_keys
	key_value_handler = {'tp':lambda x:float(x),
						 'ISI':lambda x:float(x),
						 'T':lambda x:float(x),
						 'iti':lambda x:float(x),
						 'dt':lambda x:float(x),
						 'reward':lambda x:float(x),
						 'penalty':lambda x:float(x),
						 'n':lambda x:int(x),
						 'external_var':external_var_handler,
						 'rt_cutoff':lambda x:float(x),
						 'distractor':lambda x:float(x),
						 'forced_non_decision_time':lambda x:float(x),
						 'prior_var_prob':array_parser,
						 'rt_measured_from_stim_end': lambda x: bool(x) if x.lower().strip() not in ['true','false'] else x.lower().strip()=='true',
						 'is_binary_confidence': lambda x: bool(x) if x.lower().strip() not in ['true','false'] else x.lower().strip()=='true',
						 'time_available_to_respond': time_available_to_respond_handler,
						 'session_parser': lambda x: eval(x),
						 'file_extension': extension_parser,
						 'time_conversion_to_seconds': lambda x:float(x),
						 'data_structure': data_structure_parser,
						 'excluded_files': lambda x: re.compile(x)}
	with open('experiment_details.txt','r') as f:
		buf = ''
		in_experiment = False
		experiment_name = None
		multiline_value = False
		for lineno,line in enumerate(f):
			line = clean_str(line)
			if line.endswith('\\'):
				if not multiline_value:
					multiline_value = True
				buf+=line.rstrip('\\')
				continue
			if multiline_value:
				line = buf+line
				buf = ''
				multiline_value = False
			if len(line)==0:
				continue
			if not in_experiment:
				if line.startswith('raw_data_dir:'):
					options['raw_data_dir'] = str(line.replace('raw_data_dir:','').strip(' \t\n\r'))
				elif line.startswith('begin experiment'):
					in_experiment = True
					experiment_name = str(line.replace('begin experiment','').strip(' \t\n\r'))
					if len(experiment_name)==0:
						raise IOError('Invalid options file. Encountered empty experiment name. Line number {0}'.format(lineno+1))
					elif not experiment_name in options['experiment_details'].keys():
						options['experiment_details'][experiment_name] = {'DecisionModel':{},'Fitter':{},'IO':{}}
					else:
						raise IOError('Invalid options file. Encountered a repeated begin experiment statement for experiment "{1}". Line number {0}'.format(lineno+1,experiment_name))
				elif line.startswith('end experiment'):
					raise IOError('Invalid options file. Encountered end experiment statement without corresponding open. Line number {0}'.format(lineno+1))
				elif line.startswith(tuple(valid_experiment_details_keys)):
					raise IOError('Invalid options file. Encountered experiment detail that does not corresponds to any experiment. Line number {0}'.format(lineno+1))
			else:
				if line.startswith('raw_data_dir:'):
					raise IOError('Invalid options file. Encountered raw_data_dir declaration inside experiment details declaration. Line number {0}'.format(lineno+1))
				elif line.startswith('begin experiment'):
					raise IOError('Invalid options file. Encountered begin experiment statement without having closed the previous experiment details section. Line number {0}'.format(lineno+1))
				elif line.startswith('end experiment'):
					closed_experiment = line.replace('end experiment','').strip(' \t\n\r')
					if closed_experiment!=experiment_name:
						raise IOError('Invalid options file. Attempting to close experiment {0} but the open experiment is {1}. Line number {2}'.format(closed_experiment,experiment_name,lineno+1))
					else:
						in_experiment = False
						experiment_name = None
				else:
					key,_,value = line.partition(':')
					value = value.strip()
					if not key in valid_experiment_details_keys:
						raise IOError('Invalid options file. Encountered invalid key {0} at line {1}. Valid key names are:\n {2}\n'.format(key,lineno+1,valid_experiment_details_keys))
					elif len(value)==0:
						raise IOError('Invalid options file. Encountered experiment detail key with empty value. Line number {0}'.format(lineno+1))
					if key in valid_decision_model_keys:
						options['experiment_details'][experiment_name]['DecisionModel'][key] = key_value_handler[key](value)
					elif key in valid_fitter_keys:
						options['experiment_details'][experiment_name]['Fitter'][key] = key_value_handler[key](value)
					elif key in valid_io_keys:
						options['experiment_details'][experiment_name]['IO'][key] = key_value_handler[key](value)
		if in_experiment:
			raise IOError('Invalid options file. Encountered end of file but experiment "{0}" was not closed'.format(experiment_name))
	if not os.path.isdir(options['raw_data_dir']):
		raise IOError('Options file supplies non existant raw_data_dir: {0}'.format(options['raw_data_dir']))
	return options

"""
The following diptst, dip and diptest functions were taken from
https://github.com/alimuldal/diptest Alistair Muldal's diptest python
package. It is a pure python implementation without the need to compile
with Cython.
The diptst is just a Python port of Alistair Muldans' _dip.c C function,
which itself is a port of [Martin Maechler's R module of the same
name](http://cran.r-project.org/web/packages/diptest/index.html), and uses a
slightly modified version of his C function for computing the dip statistic.
"""

def diptst(x,full_output=False, min_is_0=True, x_is_sorted=False, debug=0):
	x = x.flatten(order='K')
	n = x.shape[0]
	if n<1:
		raise ValueError('n must be >= 1')
	if not x_is_sorted:
		# force a copy to prevent inplace modification of input
		x = x.copy()
		# sort inplace
		x.sort()
	# cast to double
	x = x.astype(np.double)
	
	# Check for all values of X identical, and for 1 <= n < 4
	# LOW contains the index of the current estimate  of the lower end
	# of the modal interval, HIGH contains the index for the upper end.
	low = 0
	high = n-1
	
	dip = 0. if min_is_0 else 1.
	if n<2 or x[-1]==x[0]:
		dip = 0. if min_is_0 else 0.5/float(n)
		if full_output:
			res_dict = {
				'xs':np.array(x),
				'n':n,
				'dip':dip,
				'lo':low,
				'hi':high,
				'xl':x[low],
				'xu':x[high],
				'gcm':None,
				'lcm':None,
				'mn':None,
				'mj':None,
			}
			return dip, res_dict
		else:
			return dip
	
	# Establish the indices   mn[1..n]  over which combination is necessary
	# for the convex MINORANT (GCM) fit.
	
	mn = np.zeros(n)
	for j in range(1,n):
		mn[j] = j - 1;
		while True:
			mnj = mn[j]
			mnmnj = mn[mnj]
			if mnj==0 or (x[j]-x[mnj])*(mnj-mnmnj)<(x[mnj]-x[mnmnj])*(j-mnj):
				break
			mn[j] = mnmnj
	
	# Establish the indices   mj[1..n]  over which combination is necessary
	# for the concave MAJORANT (LCM) fit.
	mj = np.zeros(n)
	mj[n-1] = n-1
	for k in reversed(range(0,n-1)):
		mj[k] = k + 1
		while True:
			mjk = mj[k]
			mjmjk = mj[mjk]
			if mjk==(n - 1) or (x[k]-x[mjk])*(mjk-mjmjk)<(x[mjk]-x[mjmjk])*(k-mjk):
				break
			mj[k] = mjmjk
	
	# ----------------------- Start the cycling. ------------------------------- 
	# Collect the change points for the GCM from HIGH to LOW.
	not_finished = True
	while not_finished:
		gcm = np.empty(n, dtype=np.int32)
		gcm[0] = high;
		i = 0
		while gcm[i]>low:
			gcm[i+1] = mn[gcm[i]]
			i+=1
		ig = l_gcm = i # l_gcm == relevant_length(GCM)
		ix = ig-1 #  ix, ig  are counters for the convex minorant.
		
		# Collect the change points for the LCM from LOW to HIGH. 
		lcm = np.empty(n, dtype=np.int32)
		lcm[0] = low
		i = 0
		while lcm[i]<high:
			lcm[i+1] = mj[lcm[i]]
			i+=1
		ih = l_lcm = i # l_lcm == relevant_length(LCM)
		iv = 1 #  iv, ih  are counters for the concave majorant.
		
		if debug:
			print("'dip': LOOP-BEGIN: 2n*D= %-8.5g  [low,high] = [%3d,%3d]"%(dip, low,high))
			if debug >= 3:
				print(" :\n gcm[0:%d] = "%(l_gcm))
				for i in range(l_gcm):
					print("%d%s"%(gcm[i], ", " if (i < l_gcm) else "\n"))
				print(" lcm[0:%d] = "%(l_lcm))
				for i in range(l_lcm):
					print("%d%s"%(lcm[i], ", " if (i < l_lcm) else "\n"))
			else: # debug <=2
				print("; l_lcm/gcm = (%2d,%2d)\n"%(l_lcm,l_gcm))
		
		#  Find the largest distance greater than 'DIP' between the GCM and
		#  the LCM from LOW to HIGH. 
		
		d = 0.
		if l_gcm != 2 or l_lcm != 2:
			if debug:
				print("  while(gcm[ix] != lcm[iv]) :%s"%("\n" if debug>=2 else " "))
			first_iter = True
			while (first_iter or gcm[ix] != lcm[iv]):
				first_iter = False
				gcmix = gcm[ix]
				lcmiv = lcm[iv]
				if (gcmix > lcmiv):
					# If the next point of either the GCM or LCM is from the LCM,
					# calculate the distance here.
					gcmi1 = gcm[ix + 1]
					dx = (lcmiv - gcmi1 + 1) - (x[lcmiv] - x[gcmi1]) * (gcmix - gcmi1)/(x[gcmix] - x[gcmi1])
					ix+=1
					if (dx >= d):
						d = dx
						ig = ix + 1
						ih = iv - 1
						if debug >= 2:
							print(" L(%d,%d)"%(ig,ih))
				else: 
					# If the next point of either the GCM or LCM is from the GCM,
					# calculate the distance here. 
					lcmiv1 = lcm[iv - 1]
					dx = (x[gcmix] - x[lcmiv1]) * (lcmiv - lcmiv1) / (x[lcmiv] - x[lcmiv1])- (gcmix - lcmiv1 - 1)
					ix-=1
					if (dx >= d):
						d = dx
						ig = ix + 1
						ih = iv
						if debug >= 2:
							print(" G(%d,%d)"%(ig,ih))
				if (ix < 0):
					ix = 0
				if (iv > l_lcm):
					iv = l_lcm
				if debug:
					if debug>=2:
						print(" --> ix = %d, iv = %d\n"%(ix,iv))
			if(debug and debug < 2):
				print("\n")
		else: # l_gcm or l_lcm == 2
			d = 0. if min_is_0 else 1.
			if debug:
				print("  ** (l_lcm,l_gcm) = (%d,%d) ==> d := %g\n"%(l_lcm, l_gcm, float(d)))
			if d<dip:
				not_finished = False
				break
		
		#     Calculate the DIPs for the current LOW and HIGH.
		if debug:
			print("  calculating dip ..")
		j_best = j_l = -1
		j_u = -1
		
		# The DIP for the convex minorant.
		dip_l = 0.
		for j in range(ig,l_gcm):
			max_t = 1.
			j_ = -1
			jb = gcm[j + 1]
			je = gcm[j]
			if je-jb>1 and x[je]!=x[jb]:
				C = (je - jb) / (x[je] - x[jb])
				for jj in range(jb,je+1):
					t = (jj - jb + 1) - (x[jj] - x[jb]) * C
					if (max_t < t):
						max_t = t
						j_ = jj
			if dip_l < max_t:
				dip_l = max_t
				j_l = j_
		
		# The DIP for the concave majorant.
		dip_u = 0.
		for j in range(ih,l_lcm):
			max_t = 1.
			j_ = -1
			jb = lcm[j]
			je = lcm[j + 1]
			if je-jb>1 and x[je]!=x[jb]:
				C = (je - jb) / (x[je] - x[jb])
				for jj in range(jb,je+1):
					t = (x[jj] - x[jb]) * C - (jj - jb - 1)
					if (max_t < t):
						max_t = t
						j_ = jj
			if dip_u < max_t:
				dip_u = max_t
				j_u = j_
		
		if debug:
			print(" (dip_l, dip_u) = (%g, %g)"%(dip_l, dip_u))
		
		# Determine the current maximum.
		if dip_u > dip_l:
			dipnew = dip_u
			j_best = j_u
		else:
			dipnew = dip_l
			j_best = j_l
		if dip<dipnew:
			dip = dipnew
			if debug:
				print(" -> new larger dip %g (j_best = %d)\n"%(dipnew, j_best))
		elif debug:
			print("\n")
		
		# --- The following if-clause is NECESSARY  (may loop infinitely otherwise)!
		# --- Martin Maechler, Statistics, ETH Zurich, July 30 1994 ---------- 
		if low == gcm[ig] and high == lcm[ih]:
			if debug:
				print("No improvement in  low = %d  nor  high = %d --> END\n"%(low, high))
			not_finished = False
		else:
			low  = gcm[ig]
			high = lcm[ih]
	
	dip /= (2*n)
	if full_output:
		res_dict = {
			'xs':np.array(x),
			'n':n,
			'dip':dip,
			'lo':low,
			'hi':high,
			'xl':x[low],
			'xu':x[high],
			'gcm':np.array(gcm[:l_gcm]),
			'lcm':np.array(lcm[:l_lcm]),
			'mn':np.array(mn),
			'mj':np.array(mj),
		}
		return dip, res_dict
	else:
		return dip

def dip(x, full_output=False, min_is_0=True, x_is_sorted=False, debug=0):
	"""
	Hartigan & Hartigan's dip statistic

	The dip statistic measures multimodality in a sample by the maximum
	difference, over all sample points, between the empirical distribution
	function, and the unimodal distribution function that minimizes that
	maximum difference.

	Arguments:
	-----------
	x:              [n,] array  containing the input data

	full_output:    boolean, see below

	min_is_0:       boolean, if True the minimum value of the test statistic is
					allowed to be zero in cases where n <= 3 or all values in x
					are identical.

	x_is_sorted:    boolean, if True x is assumed to already be sorted in
					ascending order

	debug:          int, 0 <= debug <= 3, print debugging messages

	Returns:
	-----------
	dip:    double, the dip statistic

	[res]:  dict, returned if full_output == True. contains the following
			fields:

			xs:     sorted input data as doubles
			n:      len(x)
			dip:    dip statistic
			lo:     indices of lower end of modal interval
			hi:     indices of upper end of modal interval
			xl:     lower end of modal interval
			xu:     upper end of modal interval
			gcm:    (last-used) indices of the greatest concave majorant
			lcm:    (last-used) indices of the least concave majorant

	Reference:
	-----------
		Hartigan, J. A., & Hartigan, P. M. (1985). The Dip Test of Unimodality.
		The Annals of Statistics.
	"""

	return diptst(x, full_output=full_output, min_is_0=min_is_0, x_is_sorted=x_is_sorted, debug=debug)

def diptest(x, min_is_0=True, boot_pval=False, n_boot=2000):
	"""
	Hartigan & Hartigan's dip test for unimodality.

	For X ~ F i.i.d., the null hypothesis is that F is a unimodal distribution.
	The alternative hypothesis is that F is multimodal (i.e. at least bimodal).
	Other than unimodality, the dip test does not assume any particular null
	distribution.

	Arguments:
	-----------
	x:          [n,] array  containing the input data

	min_is_0:   boolean, see docstring for dip()

	boot_pval:  if True the p-value is computed using bootstrap samples from a
				uniform distribution, otherwise it is computed via linear
				interpolation of the tabulated critical values in dip_crit.txt.

	n_boot:     if boot_pval=True, this sets the number of bootstrap samples to
				use for computing the p-value.

	Returns:
	-----------
	dip:    double, the dip statistic

	pval:   double, the p-value for the test

	Reference:
	-----------
		Hartigan, J. A., & Hartigan, P. M. (1985). The Dip Test of Unimodality.
		The Annals of Statistics.

	"""
	n = x.shape[0]
	D = dip(x, full_output=False, min_is_0=min_is_0)

	if n <= 3:
		warnings.warn('Dip test is not valid for n <= 3')
		pval = 1.0

	elif boot_pval:

		# random uniform vectors
		boot_x = np.random.rand(n_boot, n)

		# faster to pre-sort
		boot_x.sort(axis=1)
		boot_D = np.empty(n_boot)

		for ii in xrange(n_boot):
			boot_D[ii] = dip(boot_x[ii], full_output=False,
							 min_is_0=min_is_0, x_is_sorted=True)

		pval = np.mean(D <= boot_D)

	else:

		i1 = N.searchsorted(n, side='left')
		i0 = i1 - 1

		# if n falls outside the range of tabulated sample sizes, use the
		# critical values for the nearest tabulated n (i.e. treat them as
		# 'asymptotic')
		i0 = max(0, i0)
		i1 = min(N.shape[0] - 1, i1)

		# interpolate on sqrt(n)
		n0, n1 = N[[i0, i1]]
		fn = float(n - n0) / (n1 - n0)
		y0 = np.sqrt(n0) * CV[i0]
		y1 = np.sqrt(n1) * CV[i1]
		sD = np.sqrt(n) * D

		pval = 1. - np.interp(sD, y0 + fn * (y1 - y0), SIG)

	return D, pval

def stats_battery():
	stats.chisquare
	stats.contingency.chi2_contingency
	stats.contingency.power_divergence
	stats.contingency.expected_freq
	stats.entropy
	stats.fisher_exact
	stats.friedmanchisquare
	stats.jarque_bera
	stats.ks_2samp
	stats.kstest
	stats.kruskal
	stats.kurtosistest
	stats.levene
	stats.linregress
	stats.mannwhitneyu
	stats.mood
	stats.morestats.bartlett
	stats.wilcoxon
	stats.skewtest
	stats.normaltest
	stats.pearsonr
	stats.spearmanr
	stats.ttest_1samp
	stats.ttest_ind
	stats.ttest_rel
	
	
	
	
	
