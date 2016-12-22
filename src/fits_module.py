#!/usr/bin/python
#-*- coding: UTF-8 -*-
"""
Package for fitting the decision model parameters to the behavioral
dataset

Defines the Fitter class that provides an interface to fit the
experimental data for the experiments defined in the fits_options.txt
file.

Author: Luciano Paz
Year: 2016
"""
from __future__ import division, print_function, absolute_import, unicode_literals

import enum, os, sys, math, scipy, pickle, warnings, json, logging, logging.config, copy, re
import scipy.signal
import numpy as np
from utils import normpdf,average_downsample,parse_details_file

try:
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		import matplotlib as mt
	from matplotlib import pyplot as plt
	from matplotlib.backends.backend_pdf import PdfPages
	import matplotlib.gridspec as gridspec
	from matplotlib.colors import LogNorm
	can_plot = True
except:
	can_plot = False
import data_io as io
from decision_model import DecisionModel
import cma

package_logger = logging.getLogger("fits_module")

options = parse_details_file()
experiment_details = options['experiment_details']
del options

def rt_confidence_likelihood(t,rt_confidence_pdf,RT,confidence):
	"""
	rt_confidence_likelihood(t,rt_confidence_pdf,RT,confidence):
	
	Static function that computes the joint probability density of a
	certain real valued response time, RT, and confidence as the linear
	interpolation of a discrete 2D array of RT-confidence probability
	densities computed at the discrete set of times held in the input
	array t and the array numpy.linspace(0,1,rt_confidence_pdf.shape[0])
	of confidences.
	
	Input:
		t: A numpy array with the times at which the probability density
			is known
		rt_confidence_pdf: A 2D numpy array of probability densities for
			the input array t and the confidences
			numpy.linspace(0,1,rt_confidence_pdf.shape[0]). The first
			axis of rt_confidence_pdf corresponds to the confidence index
			and the second axis corresponds to the time t.
		RT: The real valued response time of which one wishes to
			compute the probability density.
		confidence: The real valued confidence of which one wishes to
			compute the probability density.
	
	Output: A float whos value is the desired probability density
	
	"""
	if RT>t[-1] or RT<t[0]:
		return 0.
	if confidence>1 or confidence<0:
		return 0.
	nC,nT = rt_confidence_pdf.shape
	confidence_array = np.linspace(0,1,nC)
	t_ind = np.interp(RT,t,np.arange(0,nT,dtype=np.float))
	c_ind = np.interp(confidence,confidence_array,np.arange(0,nC,dtype=np.float))
	
	floor_t_ind = int(np.floor(t_ind))
	ceil_t_ind = int(np.ceil(t_ind))
	t_weight = 1.-t_ind%1.
	if floor_t_ind==nT-1:
		ceil_t_ind = floor_t_ind
		t_weight = np.array([1.])
	else:
		t_weight = np.array([1.-t_ind%1.,t_ind%1.])
	
	floor_c_ind = int(np.floor(c_ind))
	ceil_c_ind = int(np.ceil(c_ind))
	if floor_c_ind==nC-1:
		ceil_c_ind = floor_c_ind
		c_weight = np.array([1.])
	else:
		c_weight = np.array([1.-c_ind%1.,c_ind%1.])
	weight = np.ones((len(c_weight),len(t_weight)))
	for index,cw in enumerate(c_weight):
		weight[index,:]*= cw
	for index,tw in enumerate(t_weight):
		weight[:,index]*= tw
	
	prob = np.sum(rt_confidence_pdf[floor_c_ind:ceil_c_ind+1,floor_t_ind:ceil_t_ind+1]*weight)
	return prob

def rt_likelihood(t,rt_pdf,RT):
	"""
	rt_likelihood(t,rt_pdf,RT)
	
	Static function that computes the probability density of a certain
	real valued response time, RT, as the linear interpolation of a
	discrete array of RT probability densities computed at the discrete
	set of times held in the input array t.
	
	Input:
		t: A numpy array with the times at which the probability density
			is known
		rt_pdf: A numpy array of probability densities for the
			input array t.
		RT: The real valued response time of which one wishes to
			compute the probability density.
	
	Output: A float whos value is the desired probability density
	
	"""
	if RT>t[-1] or RT<t[0]:
		return 0.
	nT = rt_pdf.shape[0]
	t_ind = np.interp(RT,t,np.arange(0,nT,dtype=np.float))
	
	floor_t_ind = int(np.floor(t_ind))
	ceil_t_ind = int(np.ceil(t_ind))
	t_weight = 1.-t_ind%1.
	if floor_t_ind==nT-1:
		ceil_t_ind = floor_t_ind
		weight = np.array([1.])
	else:
		weight = np.array([1.-t_ind%1.,t_ind%1.])
	
	prob = np.sum(rt_pdf[floor_t_ind:ceil_t_ind+1]*weight)
	return prob

def confidence_likelihood(confidence_pdf,confidence):
	"""
	confidence_likelihood(confidence_pdf,confidence)
	
	Static function that computes the probability density of a certain
	real valued confidence as the linear interpolation of a discrete
	array of confidence probability densities computed at a discrete set
	of confidence values.
	
	Input:
		confidence_pdf: The array of discrete pdfs for confidences equal
			to numpy.linspace(0,1,len(confidence_pdf))
		confidence: The real valued confidence of which one wishes to
			compute the probability density.
	
	Output: A float whos value is the desired probability density
	
	"""
	if confidence>1. or confidence<0.:
		return 0.
	nC = confidence_pdf.shape[0]
	confidence_array = np.linspace(0.,1.,nC)
	c_ind = np.interp(confidence,confidence_array,np.arange(0,nC,dtype=np.float))
	
	floor_c_ind = int(np.floor(c_ind))
	ceil_c_ind = int(np.ceil(c_ind))
	if floor_c_ind==nC-1:
		ceil_c_ind = floor_c_ind
		weight = np.array([1.])
	else:
		weight = np.array([1.-c_ind%1.,c_ind%1.])
	
	prob = np.sum(confidence_pdf[floor_c_ind:ceil_c_ind+1]*weight)
	return prob

def load_Fitter_from_file(fname):
	"""
	load_Fitter_from_file(fname)
	
	Return the Fitter instance that is stored in the file with name fname.
	
	"""
	f = open(fname,'r')
	fitter = pickle.load(f)
	f.close()
	return fitter

def Fitter_filename(experiment,method,name,session,optimizer,suffix,confidence_map_method='log_odds',fits_path='fits_cognition'):
	"""
	Fitter_filename(experiment,method,name,session,optimizer,suffix,confidence_map_method='log_odds',fits_path='fits_cognition')
	
	Returns a string. Returns the formated filename for the supplied
	experiment, method, name, session, optimizer, suffix and
	confidence_map_method strings.
	
	The output has two formats for backwards compatibility:
	If confidence_map_method is 'log_odds' the output is
	os.path.join('{fits_path}','{experiment}_fit_{method}_subject_{name}_session_{session}_{optimizer}{suffix}.pkl')
	
	If confidence_map_method is not 'log_odds' the output is
	os.path.join('{fits_path}','{experiment}_fit_{method}_subject_{name}_session_{session}_{optimizer}_cmapmeth_{confidence_map_method}{suffix}.pkl')
	
	"""
	if confidence_map_method=='log_odds':
		return os.path.join(fits_path,'{experiment}_fit_{method}_subject_{name}_session_{session}_{optimizer}{suffix}.pkl'.format(
				experiment=experiment,method=method,name=name,session=session,optimizer=optimizer,suffix=suffix))
	else:
		return os.path.join(fits_path,'{experiment}_fit_{method}_subject_{name}_session_{session}_{optimizer}_cmapmeth_{confidence_map_method}{suffix}.pkl'.format(
				experiment=experiment,method=method,name=name,session=session,optimizer=optimizer,suffix=suffix,confidence_map_method=confidence_map_method))

class Fitter:
	#~ __module__ = os.path.splitext(os.path.basename(__file__))[0]
	# Initer
	def __init__(self,subjectSession,method='full_confidence',optimizer='cma',\
				decisionModelKwArgs={},suffix='',rt_cutoff=None,confidence_partition=100,\
				confidence_mapping_method='log_odds',binary_split_method='median',
				fits_path='fits_cognition'):
		"""
		Fitter(subjectSession,method='full_confidence',optimizer='cma',\
				decisionModelKwArgs={},suffix='',rt_cutoff=None,confidence_partition=100,\
				confidence_mapping_method='log_odds',binary_split_method='median',
				fits_path='fits_cognition')
		
		Construct a Fitter instance that interfaces the fitting procedure
		of the model's likelihood of the observed subjectSession data.
		Input:
			subjectSession: A SubjectSession instance from data_io_cognition
				package
			method: Specifies the merit function to use for the fits. Can
				be 'full' (likelihood of the join RT and performance data),
				'confidence_only' (likelihood of the confidence),
				'full_confidence' (likelihood of the joint RT, performance
				and confidence data), 'binary_confidence_only' (likelihood
				of joint RT and confidence above or below the median confidence)
				or 'full_binary_confidence' (likelihood of joint RT,
				performance and confidence above or below the median confidence).
			optimizer: The optimizer used for the fitting procedure.
				Available optimizers are 'cma', scipy.optimize.basinhopping
				and scipy.optimize methods called from minimize and
				minimize_scalar.
			decisionModelKwArgs: A dict of kwargs to use for the construction
				of the used DecisionModel instance
			suffix: A string suffix to append to the saved filenames
			rt_cutoff: None or a float. If not None, it is used to
				override the experiment default rt_cutoff specified in
				the fits_options.txt file.
			confidence_partition: The number of bins used to compute the
				confidence histograms
			confidence_mapping_method: A string that identifies the
				high confidence mapping method. Available values are
				'log_odds' and 'belief'. If 'log_odds' the confidence
				mapping is the composition of the log odds of the decision
				bounds in g space (see cost_time DecisionModel decision
				bounds) with a sigmoid. If 'belief', the confidence mapping
				is a linear mapping of the rescaled decision bounds in g
				space (the rescaling makes high confidence values (1 for
				hits and 0 for misses) equal to 1 and g=0.5 equal to 0).
				This linear mapping is also clipped to the interval [0,1].
			binary_split_method: A string that identifies the split
				method to binarize the subjectSession's confidence
				reports. This is only used when calling the binary
				confidence merit functions. To find the available split
				methods refer to self.get_binary_confidence.
			fits_path: The path to the directory where the fit results
				should be saved.
		
		Output: A Fitter instance that can be used as an interface to
			fit the model's parameters to the subjectSession data
		
		Example assuming that subjectSession is a data_io_cognition.SubjectSession
		instance:
		fitter = Fitter(subjectSession)
		fitter.fit()
		fitter.save()
		print(fitter.stats)
		fitter.plot()
		
		"""
		package_logger.debug('Creating Fitter instance for "{experiment}" experiment and "{name}" subject with sessions={session}'.format(
						experiment=subjectSession.experiment,name=subjectSession.name,session=subjectSession.session))
		self.logger = logging.getLogger("fits_module.Fitter")
		self.decisionModelKwArgs = decisionModelKwArgs.copy()
		self.logger.debug('Setted decisionModelKwArgs = {0}'.format(self.decisionModelKwArgs))
		self.fits_path = fits_path
		self.logger.debug('Setted fits_path = %s',self.fits_path)
		self.set_experiment(subjectSession.experiment)
		if not rt_cutoff is None:
			self.rt_cutoff = float(rt_cutoff)
		self.set_subjectSession_data(subjectSession)
		self.method = str(method)
		self.logger.debug('Setted Fitter method = %s',self.method)
		self.optimizer = str(optimizer)
		self.logger.debug('Setted Fitter optimizer = %s',self.optimizer)
		self.suffix = str(suffix)
		self.logger.debug('Setted Fitter suffix = %s',self.suffix)
		self.confidence_partition = int(confidence_partition)
		self.confidence_mapping_method = str(confidence_mapping_method).lower()
		self.binary_split_method = str(binary_split_method).lower()
		self.__fit_internals__ = None
	
	def __str__(self):
		if hasattr(self,'_fit_arguments'):
			_fit_arguments = self._fit_arguments
		else:
			_fit_arguments = None
		if hasattr(self,'_fit_output'):
			_fit_output = self._fit_output
		else:
			_fit_output = None
		string = """
<{class_module}.{class_name} object at {address}>
fits_path = {fits_path},
experiment = {experiment},
method = {method},
optimizer = {optimizer},
confidence_mapping_method = {confidence_mapping_method},
suffix = {suffix},
binary_split_method = {binary_split_method},
rt_cutoff = {rt_cutoff},
decisionModelKwArgs = {decisionModelKwArgs},
confidence_partition = {confidence_partition},
_fit_arguments = {_fit_arguments},
_fit_output = {_fit_output}
		""".format(class_module=self.__class__.__module__,
					class_name=self.__class__.__name__,
					address=hex(id(self)),
					fits_path=self.fits_path,
					experiment = self.experiment,
					method = self.method,
					optimizer = self.optimizer,
					confidence_mapping_method = self.confidence_mapping_method,
					suffix = self.suffix,
					binary_split_method = self.binary_split_method,
					rt_cutoff = self.rt_cutoff,
					decisionModelKwArgs = self.decisionModelKwArgs,
					confidence_partition = self.confidence_partition,
					_fit_arguments = self._fit_arguments,
					_fit_output = self._fit_output)
		return string
	
	# Setters
	def set_experiment(self,experiment):
		"""
		self.set_experiment(self)
		
		Set the Fitter's experiment
		"""
		self.experiment = str(experiment)
		try:
			decisionModelKwArgs = experiment_details[self.experiment]['DecisionModel'].copy()
			decisionModelKwArgs.update(self.decisionModelKwArgs)
			self.logger.debug('Will construct DecisionModel instance with the folling kwargs:\n{0}'.format(decisionModelKwArgs))
			self.dm = DecisionModel(**decisionModelKwArgs)
			if 'time_available_to_respond' in experiment_details[self.experiment]['Fitter']:
				self._time_available_to_respond = experiment_details[self.experiment]['Fitter']['time_available_to_respond']
			else:
				self._time_available_to_respond = np.inf
			self.logger.debug('Set time_available_to_respond = {0}'.format(self._time_available_to_respond))
			if np.isinf(self._time_available_to_respond):
				if 'rt_cutoff' in experiment_details[self.experiment]['Fitter']:
					self.rt_cutoff = experiment_details[self.experiment]['Fitter']['rt_cutoff']
				else:
					self.rt_cutoff = 14.
			else:
				self.rt_cutoff = self._time_available_to_respond
			self.logger.debug('Set rt_cutoff = {0}'.format(self.rt_cutoff))
			if 'distractor' in experiment_details[self.experiment]['Fitter']:
				self._distractor = experiment_details[self.experiment]['Fitter']['distractor']
			else:
				self._distractor = 0.
			self.logger.debug('Set distractor = {0}'.format(self._distractor))
			if 'forced_non_decision_time' in experiment_details[self.experiment]['Fitter']:
				self._forced_non_decision_time = experiment_details[self.experiment]['Fitter']['forced_non_decision_time']
			else:
				self._forced_non_decision_time = 0.
			self.logger.debug('Set forced_non_decision_time = {0}'.format(self._forced_non_decision_time))
			self._ISI = experiment_details[self.experiment]['Fitter']['ISI']
			self.logger.debug('Set ISI = {0}'.format(self._ISI))
			if 'rt_measured_from_stim_end' in experiment_details[self.experiment]['Fitter']:
				self._rt_measured_from_stim_end = experiment_details[self.experiment]['Fitter']['rt_measured_from_stim_end']
			else:
				self._rt_measured_from_stim_end = False
			self.logger.debug('Set rt_measured_from_stim_end = {0}'.format(self._rt_measured_from_stim_end))
			if 'is_binary_confidence' in experiment_details[self.experiment]['Fitter']:
				self._is_binary_confidence = experiment_details[self.experiment]['Fitter']['is_binary_confidence']
			else:
				self._is_binary_confidence = False
			self.logger.debug('Set is_binary_confidence = {0}'.format(self._is_binary_confidence))
		except KeyError:
			raise ValueError('Experiment {0} not specified in fits_options.txt.\nPlease specify the experiment details before attempting to fit its data.'.format(self.experiment))
		self.logger.debug('Setted Fitter experiment = %s',self.experiment)
	
	def set_subjectSession_data(self,subjectSession):
		"""
		self.set_subjectSession_data(subjectSession)
		
		This function loads the data from the subjectSession and stores
		the relevant information for the fitting process, whice are the:
		response times, performance, confidence and observed contrasts.
		This function also extracts the prior contrast distribution
		variance, the min RT and the max RT that are used during the
		fitting process.
		
		"""
		self.subjectSession = subjectSession
		self._subjectSession_state = subjectSession.__getstate__()
		self.logger.debug('Setted Fitter _subjectSession_state')
		self.logger.debug('experiment:%(experiment)s, name:%(name)s, session:%(session)s, data_dir:%(data_dir)s',self._subjectSession_state)
		dat = subjectSession.load_data()
		self.logger.debug('Loading subjectSession data')
		self.rt = dat['rt']
		self.rt+=self._forced_non_decision_time
		valid_trials = self.rt<=self.rt_cutoff
		self.rt = self.rt[valid_trials]
		self.max_RT = np.max(self.rt)
		self.min_RT = np.min(self.rt)
		dat = dat[valid_trials]
		
		trials = len(self.rt)
		if trials==0:
			raise RuntimeError('No trials can be fitted')
		
		self.contrast = (dat['contrast']-self._distractor)/self._ISI
		self.performance = dat['performance']
		self.confidence = dat['confidence']
		if not self.dm.known_variance():
			try:
				self.external_var = dat['variance']/self._ISI
			except:
				raise RuntimeError('Cannot perform fits for unknown variance DecisionModel if the data does not have a "variance" field')
			self.unique_stim,self.stim_indeces,counts = utils.unique_rows(np.array([self.contrast,self.external_var]).T,return_inverse=True,return_counts=True)
			self.stim_probs = count.astype(np.float64)/np.sum(count.astype(np.float64))
		self.logger.debug('Trials loaded = %d',len(self.performance))
		self.mu,self.mu_indeces,count = np.unique(self.contrast,return_inverse=True,return_counts=True)
		self.logger.debug('Number of different drifts = %d',len(self.mu))
		self.mu_prob = count.astype(np.float64)/np.sum(count.astype(np.float64))
		if self.mu[0]==0:
			mus = np.concatenate((-self.mu[-1:0:-1],self.mu))
			p = np.concatenate((self.mu_prob[-1:0:-1],self.mu_prob))
			p[mus!=0]*=0.5
		else:
			mus = np.concatenate((-self.mu[::-1],self.mu))
			p = np.concatenate((self.mu_prob[::-1],self.mu_prob))*0.5
		
		self._prior_mu_var = np.sum(p*(mus-np.sum(p*mus))**2)
		self.logger.debug('Setted Fitter _prior_mu_var = %f',self._prior_mu_var)
	
	def __setstate__(self,state):
		"""
		self.__setstate__(state)
		
		Only used when loading from a pickle file to init the Fitter
		instance. Could also be used to copy one Fitter instance to
		another one.
		
		"""
		self.logger = logging.getLogger("fits_module.Fitter")
		self.decisionModelKwArgs = state['decisionModelKwArgs']
		self.set_experiment(state['experiment'])
		self.rt_cutoff = float(state['rt_cutoff'])
		if 'fits_path' in state.keys():
			self.fits_path = state['fits_path']
		else:
			self.fits_path = 'fits_cognition'
		if 'confidence_partition' in state.keys():
			self.confidence_partition = int(state['confidence_partition'])
		else:
			self.confidence_partition = 100.
		self.method = state['method']
		self.optimizer = state['optimizer']
		self.suffix = state['suffix']
		self.set_subjectSession_data(io.SubjectSession(name=state['subjectSession_state']['name'],
													   session=state['subjectSession_state']['session'],
													   experiment=self.experiment,
													   data_dir=state['subjectSession_state']['data_dir']))
		if '_start_point' in state.keys():
			self._start_point = state['_start_point']
		if '_bounds' in state.keys():
			self._bounds = state['_bounds']
		if '_fitted_parameters' in state.keys():
			self._fitted_parameters = state['_fitted_parameters']
		if '_fixed_parameters' in state.keys():
			self._fixed_parameters = state['_fixed_parameters']
		if 'fit_arguments' in state.keys():
			self.set_fit_arguments(state['fit_arguments'])
		if 'fit_output' in state.keys():
			self._fit_output = state['fit_output']
			# Bug fix for old version Fitter instances in which the fit_output tuple was constructed incorrectly for the cma method
			if isinstance(self._fit_output[1],tuple):
				self._fit_output = (self._fit_output[0],)+self._fit_output[1]
		if 'confidence_mapping_method' in state.keys():
			self.confidence_mapping_method = state['confidence_mapping_method']
		else:
			self.confidence_mapping_method = 'log_odds'
		if 'binary_split_method' in state.keys():
			self.binary_split_method = state['binary_split_method']
		else:
			self.binary_split_method = 'median'
		self.__fit_internals__ = None
	
	def set_fixed_parameters(self,fixed_parameters={}):
		"""
		self.set_fixed_parameters(fixed_parameters={})
		
		Set the fixed_parameters by merging the supplied fixed_parameters
		input dict with the default fixed_parameters. Note that these
		fixed_parameters need to be sanitized before being used to init
		the minimizer. Also note that this method sets the unsanitized fitted
		parameters as the complement of the fixed_parameters keys.
		
		Input:
			fixed_parameters: A dict whose keys are the parameter names and
				the values are the corresponding parameter fixed values
		
		"""
		defaults = self.default_fixed_parameters()
		defaults.update(fixed_parameters)
		fittable_parameters = self.get_fittable_parameters()
		self._fixed_parameters = fixed_parameters.copy()
		self._fitted_parameters = []
		for par in fittable_parameters:
			if par not in self._fixed_parameters.keys():
				self._fitted_parameters.append(par)
		self.logger.debug('Setted Fitter fixed_parameters = %s',self._fixed_parameters)
		self.logger.debug('Setted Fitter fitted_parameters = %s',self._fitted_parameters)
	
	def set_start_point(self,start_point={}):
		"""
		self.set_start_point(start_point={})
		
		Set the start_point by merging the supplied start_point input dict with
		the default start_point. Note that this start_point need to be sanitized
		before being used to init the minimizer.
		
		Input:
			start_point: A dict whose keys are the parameter names and
				the values are the corresponding parameter starting value
		
		"""
		defaults = self.default_start_point()
		defaults.update(start_point)
		self._start_point = defaults
		self.logger.debug('Setted Fitter start_point = %s',self._start_point)
	
	def set_bounds(self,bounds={}):
		"""
		self.set_bounds(bounds={})
		
		Set the bounds by merging the supplied bounds input dict with
		the default bounds. Note that these bounds need to be sanitized
		before being used to init the minimizer.
		
		Input:
			bounds: A dict whose keys are the parameter names and
				the values are a list with the [low_bound,up_bound] values.
		
		"""
		defaults = self.default_bounds()
		defaults.update(bounds)
		self._bounds = defaults
		self.logger.debug('Setted Fitter bounds = %s',self._bounds)
	
	def set_optimizer_kwargs(self,optimizer_kwargs={}):
		"""
		self.set_optimizer_kwargs(optimizer_kwargs={})
		
		Set the optimizer_kwargs by merging the supplied optimizer_kwargs
		input dict with the default optimizer_kwargs. Note that these
		kwargs do not need any further sanitation that could depend on
		the Fitter's method.
		
		"""
		defaults = self.default_optimizer_kwargs()
		defaults.update(optimizer_kwargs)
		self.optimizer_kwargs = defaults
		self.logger.debug('Setted Fitter optimizer_kwargs = %s',self.optimizer_kwargs)
	
	def set_fit_arguments(self,fit_arguments):
		"""
		self.set_fit_arguments(fit_arguments)
		
		Set the instance's fit_arguments and sanitized: start point, bounds
		optimizer_kwargs, fitted parameters and fixed parameters.
		
		Input:
		fit_argument: A dict with keys: start_point, bounds, optimizer_kwargs
			fitted_parameters and fixed_parameters
		
		"""
		self._fit_arguments = fit_arguments
		self.start_point = fit_arguments['start_point']
		self.bounds = fit_arguments['bounds']
		self.optimizer_kwargs = fit_arguments['optimizer_kwargs']
		self.fitted_parameters = fit_arguments['fitted_parameters']
		self.fixed_parameters = fit_arguments['fixed_parameters']
	
	# Getters
	def get_parameters_dict(self):
		"""
		self.get_parameters_dict():
		
		Get a dict with all the model's parameter names as keys. The
		the values of the fixed parameters are taken from 
		self.get_fixed_parameters() and the rest of the parameter values
		are taken from the self.get_start_point() method.
		
		"""
		parameters = self.get_fixed_parameters().copy()
		start_point = self.get_start_point()
		for fp in self.get_fitted_parameters():
			parameters[fp] = start_point[fp]
		return parameters
	
	def get_parameters_dict_from_array(self,x):
		"""
		self.get_parameters_dict_from_array(x):
		
		Get a dict with all the model's parameter names as keys. The
		the values of the fixed parameters are taken from the 
		sanitized fixed parameters and the fitted parameters' values
		are taken from the supplied array. The array elements must have
		the same order as the sanitized fitted parameters.
		
		"""
		parameters = self.fixed_parameters.copy()
		try:
			for index,key in enumerate(self.fitted_parameters):
				parameters[key] = x[index]
		except IndexError:
			parameters[self.fitted_parameters[0]] = x
		return parameters
	
	def get_parameters_dict_from_fit_output(self,fit_output=None):
		"""
		self.get_parameters_dict_from_fit_output(fit_output=None):
		
		Get a dict with all the model's parameter names as keys. The
		the values of the fixed parameters are taken from the 
		sanitized fixed parameters and the fitted parameters' values
		are taken from the fit_output. If fit_output is None, the
		instance's fit_output is used instead.
		
		"""
		if fit_output is None:
			fit_output = self._fit_output
		parameters = self.fixed_parameters.copy()
		parameters.update(fit_output[0])
		return parameters
	
	def get_fixed_parameters(self):
		"""
		self.get_fixed_parameters()
		
		This function first attempts to return the sanitized fixed parameters.
		If this fails (most likely because the fixed parameters were not sanitized)
		it attempts to return the setted fixed parameters.
		If this fails, because the fixed parameters were not set, it returns the
		default fixed parameters.
		This function always returns a dict that has the parameter names
		as keys and as values floats with the parameter's fixed value.
		
		"""
		try:
			return self.fixed_parameters
		except:
			try:
				return self._fixed_parameters
			except:
				return self.default_fixed_parameters()
	
	def get_fitted_parameters(self):
		"""
		self.get_fitted_parameters()
		
		This function first attempts to return the sanitized fitted parameters.
		If this fails (most likely because the fitted parameters were not sanitized)
		it attempts to return the setted fitted parameters.
		If this fails, because the fitted parameters were not set, it returns the
		default fitted parameters.
		This function always returns a list of parameter names.
		
		"""
		try:
			return self.fitted_parameters
		except:
			try:
				return self._fitted_parameters
			except:
				return [p for p in self.get_fittable_parameters() if p not in self.default_fixed_parameters().keys()]
	
	def get_start_point(self):
		"""
		self.get_start_point()
		
		This function first attempts to return the sanitized start point, which
		is an array of shape (2,len(self.fitted_parameters)).
		If this fails (most likely because the start point was not sanitized)
		it attempts to return the setted parameter start point dictionary.
		This dict has the parameter names as keys and as values floats
		with the parameter's value.
		If this fails, because the start point was not set, it returns the
		default start point.
		
		"""
		try:
			return self.start_point
		except:
			try:
				return self._start_point
			except:
				return self.default_start_point()
	
	def get_bounds(self):
		"""
		self.get_bounds()
		
		This function first attempts to return the sanitized bounds, which
		is an array of shape (2,len(self.fitted_parameters)).
		If this fails (most likely because the bounds were not sanitized)
		it attempts to return the setted parameter bound dictionary.
		This dict has the parameter names as keys and as values lists
		with [low_bound,high_bound] values.
		If this fails, because the bounds were not set, it returns the
		default bounds.
		
		"""
		try:
			return self.bounds
		except:
			try:
				return self._bounds
			except:
				return self.default_bounds()
	
	def __getstate__(self):
		"""
		self.__getstate__()
		
		Get the Fitter instance's state dictionary. This function is used
		when pickling the Fitter.
		"""
		state = {'experiment':self.experiment,
				 'rt_cutoff':self.rt_cutoff,
				 'subjectSession_state':self._subjectSession_state,
				 'method':self.method,
				 'optimizer':self.optimizer,
				 'suffix':self.suffix,
				 'decisionModelKwArgs':self.decisionModelKwArgs,
				 'confidence_partition':self.confidence_partition,
				 'confidence_mapping_method':self.confidence_mapping_method,
				 'binary_split_method':self.binary_split_method,
				 'fits_path':self.fits_path}
		if hasattr(self,'_start_point'):
			state['_start_point'] = self._start_point
		if hasattr(self,'_bounds'):
			state['_bounds'] = self._bounds
		if hasattr(self,'_fitted_parameters'):
			state['_fitted_parameters'] = self._fitted_parameters
		if hasattr(self,'_fixed_parameters'):
			state['_fixed_parameters'] = self._fixed_parameters
		if hasattr(self,'_fit_arguments'):
			state['fit_arguments'] = self._fit_arguments
		if hasattr(self,'_fit_output'):
			state['fit_output'] = self._fit_output
		return state
	
	def get_fittable_parameters(self):
		"""
		self.get_fittable_parameters()
		
		Returns self.get_decision_parameters()+self.get_confidence_parameters()
		"""
		return self.get_decision_parameters()+self.get_confidence_parameters()
	
	def get_decision_parameters(self):
		"""
		self.get_decision_parameters()
		
		Returns a list of the model's decision parameters. In the current version:
		['cost','dead_time','dead_time_sigma','phase_out_prob','internal_var']
		
		"""
		return ['cost','dead_time','dead_time_sigma','phase_out_prob','internal_var']
	
	def get_confidence_parameters(self):
		"""
		self.get_confidence_parameters()
		
		Returns a list of the model's confidence parameters. In the current version:
		['high_confidence_threshold','confidence_map_slope']
		
		"""
		return ['high_confidence_threshold','confidence_map_slope']
	
	def get_dead_time_convolver(self,parameters):
		"""
		self.get_dead_time_convolver(parameters):
		
		This function returns the dead time (aka non-decision time) distribution,
		which is convolved with the first passage time probability
		density to get the real response time distribution.
		
		Input:
			parameters: A dict with keys 'dead_time' and 'dead_time_sigma'
				that hold the values of said parameters
		
		Output:
			(conv_val,conv_x)
			conv_val is an array with the values of the dead time
			distribution for the times that are in conv_x.
		
		"""
		return self.dm.get_dead_time_convolver(parameters['dead_time'],parameters['dead_time_sigma'])
	
	def get_key(self,merge=None):
		"""
		self.get_key(merge=None)
		
		This function returns a string that can be used as a Fitter_plot_handler's
		key.
		The returned key depends on the merge input as follows.
		If merge=None: key={self.experiment}_subject_{self.subjectSession.get_name()}_session_{self.subjectSession.get_session()}
		If merge='subjects': key={self.experiment}_session_{self.subjectSession.get_session()}
		If merge='sessions': key={self.experiment}_subject_{self.subjectSession.get_name()}
		If merge='all': key={self.experiment}
		
		"""
		experiment = self.experiment
		subject = self.subjectSession.get_name()
		session = self.subjectSession.get_session()
		if merge is None:
			key = "{experiment}_subject_{subject}_session_{session}".format(experiment=experiment,
					subject=subject,session=session)
		elif merge=='subjects':
			key = "{experiment}_session_{session}".format(experiment=experiment,session=session)
		elif merge=='sessions':
			key = "{experiment}_subject_{subject}".format(experiment=experiment,subject=subject)
		elif merge=='all':
			key = "{experiment}".format(experiment=experiment)
		else:
			raise ValueError('Unknown merge option {0}. Available values are None, "subjects", "sessions" and "all"'.format(merge))
		return key
	
	def get_fitter_plot_handler(self,edges=None,merge=None,fit_output=None):
		"""
		self.get_fitter_plot_handler(edges=None,merge=None,fit_output=None)
		
		This function returns a Fitter_plot_handler instance that is
		constructed using the Fitter instance's subjectSession
		performance, response time and confidence data, and the
		output of self.theoretical_rt_confidence_distribution(fit_output)
		
		Input:
			edges: A numpy array of edges used to construct the
				subjectSession's response time histogram. If it is None,
				edges = np.linspace(0,self.rt_cutoff,51)
			merge: Can be None or a str in ['all','sessions','subjects'].
				This is used to set the Fitter_plot_handler's key so
				that when it is added to another Fitter_plot_handler
				instance the data with the same key is properly merged.
				Refer to self.get_key() for more details on the used key.
			fit_output: Can be None or a tuple like the one returned by
				self.sanitize_fmin_output. It is used in the call to
				self.theoretical_rt_confidence_distribution and can be
				used to specify another set of parameters to use to
				compute the theoretical distribution. If it is None,
				then the Fitter instance's _fit_output attribute is used.
		
		"""
		if edges is None:
			edges = np.linspace(0,self.rt_cutoff,51)
		rt_edges = edges
		rt_centers = np.array([0.5*(e1+e0) for e1,e0 in zip(rt_edges[1:],rt_edges[:-1])])
		c_edges = np.linspace(0,1,self.confidence_partition+1)
		c_centers = np.array([0.5*(e1+e0) for e1,e0 in zip(c_edges[1:],c_edges[:-1])])
		dt = rt_edges[1]-rt_edges[0]
		dc = c_edges[1]-c_edges[0]
		hit = self.performance==1
		miss = np.logical_not(hit)
		subject_hit_histogram2d = np.histogram2d(self.rt[hit], self.confidence[hit], bins=[rt_edges,c_edges])[0].astype(np.float).T/dt
		subject_miss_histogram2d = np.histogram2d(self.rt[miss], self.confidence[miss], bins=[rt_edges,c_edges])[0].astype(np.float).T/dt
		
		subject_rt = np.array([np.sum(subject_hit_histogram2d,axis=0),
							   np.sum(subject_miss_histogram2d,axis=0)])
		subject_confidence = np.array([np.sum(subject_hit_histogram2d,axis=1),
									   np.sum(subject_miss_histogram2d,axis=1)])*dt
		
		self.set_fixed_parameters()
		model,t = self.theoretical_rt_confidence_distribution(fit_output)
		binary_confidence = self.method=='binary_confidence_only' or self.method=='full_binary_confidence'
		if binary_confidence:
			c = np.array([0,1])
		else:
			c = np.linspace(0,1,self.confidence_partition)
		model*=len(self.performance)
		model_hit_histogram2d = model[0]
		model_miss_histogram2d = model[1]
		model_rt = np.sum(model,axis=1)
		model_confidence = np.sum(model,axis=2)*self.dm.dt
		
		key = self.get_key(merge)
		output = {key:{'experimental':{'hit_histogram':subject_hit_histogram2d,
									   'miss_histogram':subject_miss_histogram2d,
									   'rt':subject_rt,
									   'confidence':subject_confidence,
									   't_array':rt_edges,
									   'c_array':c_edges},
						'theoretical':{'hit_histogram':model_hit_histogram2d,
									   'miss_histogram':model_miss_histogram2d,
									   'rt':model_rt,
									   'confidence':model_confidence,
									   't_array':t,
									   'c_array':c}}}
		return Fitter_plot_handler(output,self.binary_split_method)
	
	def get_save_file_name(self):
		"""
		self.get_save_file_name()
		
		An alias for the package's function Fitter_filename. This method
		simply returns the output of the call:
		Fitter_filename(experiment=self.experiment,method=self.method,
				name=self.subjectSession.get_name(),
				session=self.subjectSession.get_session(),
				optimizer=self.optimizer,suffix=self.suffix,
				confidence_map_method=self.confidence_mapping_method)
		
		"""
		return Fitter_filename(experiment=self.experiment,method=self.method,name=self.subjectSession.get_name(),
				session=self.subjectSession.get_session(),optimizer=self.optimizer,suffix=self.suffix,
				confidence_map_method=self.confidence_mapping_method,fits_path=self.fits_path)
	
	def get_jacobian_dx(self):
		"""
		self.get_jacobian_dx()
		
		This function returns a dict where the keys are fittable parameter
		names and the values are the parameter displacements that should
		be used to compute the numerical jacobian.
		Regretably, we cannot compute an analytical form of the derivative
		of the first passage time probability density in decision_model.DecisionModel.rt
		and this forces us to use a numerical approximation of the
		jacobian. The values returned by this function are only used
		with the scipy's optimize methods: CG, BFGS, Newton-CG, L-BFGS-B,
		TNC, SLSQP, dogleg and trust-ncg.
		
		"""
		jac_dx = {'cost':1e-5,
				  'internal_var':1e-4,
				  'dead_time':1e-6,
				  'dead_time_sigma':1e-6,
				  'phase_out_prob':1e-6,
				  'high_confidence_threshold':8e-4,
				  'confidence_map_slope':1e-4}
		return jac_dx
	
	def get_binary_confidence_reports(self):
		"""
		self.get_binary_confidence_reports()
		
		Returns the subjectSession's binarized confidence reports
		according to the binary_split_method attribute. Currently only 3
		methods are available: 'median', 'half' and 'mean'. These methods
		determine the split value and every confidence report that is
		below said split is converted to 0 (low confidence) and those
		greater or equal are converted to 1 (high confidence).
		If binary_split_method is median, the split value is the median
		of the confidence reports.
		If binary_split_method is half, the split value is 0.5.
		If binary_split_method is mean, the split value is the mean of
		the confidence reports.
		
		"""
		if self._is_binary_confidence:
			return self.confidence
		else:
			if self.binary_split_method=='median':
				split = np.median(self.confidence)
			elif self.binary_split_method=='half':
				split = 0.5
			elif self.binary_split_method=='mean':
				split = np.mean(self.confidence)
			else:
				raise NotImplementedError("The split method {0} is not implemented".format(split_method))
			binary_confidence = np.ones_like(self.confidence)
			binary_confidence[self.confidence<split] = 0
			return binary_confidence
	
	# Defaults
	
	def default_fixed_parameters(self):
		"""
		self.default_fixed_parameters()
		
		Returns the default parameter fixed_parameters. By default no
		parameter is fixed so it simply returns an empty dict.
		
		"""
		return {}
	
	def default_start_point(self,forceCompute=False):
		"""
		self.default_start_point()
		
		Returns the default parameter start_point that depend on
		the subjectSession's response time distribution and
		performance. This function is very fine tuned to get a good
		starting point for every parameter automatically.
		This function returns a dict with the parameter names as keys and
		the corresponding start_point floating point as values.
		
		"""
		try:
			if forceCompute:
				self.logger.debug('Forcing default start point recompute')
				raise Exception('Forcing recompute')
			return self.__default_start_point__
		except:
			if hasattr(self,'_fixed_parameters'):
				self.__default_start_point__ = self._fixed_parameters.copy()
			else:
				self.__default_start_point__ = {}
			if not 'cost' in self.__default_start_point__.keys() or self.__default_start_point__['cost'] is None:
				self.__default_start_point__['cost'] = 0.02
			if not 'internal_var' in self.__default_start_point__.keys() or self.__default_start_point__['internal_var'] is None:
				try:
					from scipy.optimize import minimize
					with warnings.catch_warnings():
						warnings.simplefilter("ignore")
						if self.dm.known_variance():
							fun = lambda a: (np.mean(self.performance)-np.sum(self.mu_prob/(1.+np.exp(-0.596*self.mu/a))))**2
						else:
							fun = lambda a: (np.mean(self.performance)-np.sum(self.stim_prob/(1.+np.exp(-0.596*self.unique_stims[:,0]/np.sqrt(self.unique_stims[:,1]+a**2)))))**2
						res = minimize(fun,1000.,method='Nelder-Mead')
					self.__default_start_point__['internal_var'] = res.x[0]**2
				except Exception as e:
					self.logger.warning('Could not fit internal_var from data')
					self.__default_start_point__['internal_var'] = 1500.
			if not 'phase_out_prob' in self.__default_start_point__.keys() or self.__default_start_point__['phase_out_prob'] is None:
				self.__default_start_point__['phase_out_prob'] = 0.05
			if not 'dead_time' in self.__default_start_point__.keys() or self.__default_start_point__['dead_time'] is None:
				dead_time = sorted(self.rt)[int(0.025*len(self.rt))]
				self.__default_start_point__['dead_time'] = dead_time
			if not 'confidence_map_slope' in self.__default_start_point__.keys() or self.__default_start_point__['confidence_map_slope'] is None:
				if self.confidence_mapping_method=='log_odds':
					self.__default_start_point__['confidence_map_slope'] = 17.2
				elif self.confidence_mapping_method=='belief':
					self.__default_start_point__['confidence_map_slope'] = 1.
				else:
					raise ValueError('Unknown confidence_mapping_method: {0}'.format(self.confidence_mapping_method))
			
			must_make_expensive_guess = ((not 'dead_time_sigma' in self.__default_start_point__.keys()) or\
										((not 'high_confidence_threshold' in self.__default_start_point__.keys()) and self.method!='full'))
			if must_make_expensive_guess:
				self.dm.set_cost(self.__default_start_point__['cost'])
				self.dm.set_internal_var(self.__default_start_point__['internal_var'])
				xub,xlb = self.dm.xbounds()
				first_passage_pdf = None
				for drift,drift_prob in zip(self.mu,self.mu_prob):
					gs = np.array(self.dm.fpt(drift,bounds=(xub,xlb)))
					if first_passage_pdf is None:
						first_passage_pdf = gs*drift_prob
					else:
						first_passage_pdf+= gs*drift_prob
				first_passage_pdf/=(np.sum(first_passage_pdf)*self.dm.dt)
				
				if not 'dead_time_sigma' in self.__default_start_point__.keys() or self.__default_start_point__['dead_time_sigma'] is None:
					mean_pdf = np.sum(first_passage_pdf*self.dm.t)*self.dm.dt
					var_pdf = np.sum(first_passage_pdf*(self.dm.t-mean_pdf)**2)*self.dm.dt
					var_rt = np.var(self.rt)
					min_dead_time_sigma_sp = 0.01
					if var_rt-var_pdf>0:
						self.__default_start_point__['dead_time_sigma'] = np.max([np.sqrt(var_rt-var_pdf),min_dead_time_sigma_sp])
					else:
						self.__default_start_point__['dead_time_sigma'] = min_dead_time_sigma_sp
				
				rt_mode_ind = np.argmax(first_passage_pdf[0])
				rt_mode_ind+= 4 if self.dm.nT-rt_mode_ind>4 else 0
				if not 'high_confidence_threshold' in self.__default_start_point__.keys() or self.__default_start_point__['high_confidence_threshold'] is None:
					if self.confidence_mapping_method=='log_odds':
						log_odds = self.dm.log_odds()
						self.__default_start_point__['high_confidence_threshold'] = log_odds[0][rt_mode_ind]
					elif self.confidence_mapping_method=='belief':
						bounds = self.dm.bounds
						self.__default_start_point__['high_confidence_threshold'] = 2*self.dm.bounds[0][rt_mode_ind]-1
			else:
				if not 'high_confidence_threshold' in self.__default_start_point__.keys() or self.__default_start_point__['high_confidence_threshold'] is None:
					self.__default_start_point__['high_confidence_threshold'] = 0.3
			
			return self.__default_start_point__
	
	def default_bounds(self):
		"""
		self.default_bounds()
		
		Returns the default parameter bounds that depend on
		whether the decision bounds are invariant or not.
		This function returns a dict with the parameter names as keys and
		the corresponding [lower_bound, upper_bound] list as values.
		
		"""
		defaults = {'cost':[0.,1.],'dead_time':[0.,1.5],'dead_time_sigma':[0.001,6.]}
		defaults['phase_out_prob'] = [0.,0.2]
		default_sp = self.default_start_point()
		defaults['internal_var'] = [default_sp['internal_var']*0.2,default_sp['internal_var']*1.8]
		invariant_decision_bounds = ('cost' in self.get_fixed_parameters().keys() and 'internal_var' in self.get_fixed_parameters().keys()) or \
									self.method in ['confidence_only','binary_confidence_only']
		if self.confidence_mapping_method=='log_odds':
			if invariant_decision_bounds:
				try:
					log_odds = self.dm.log_odds()
					mb = np.min(log_odds)
					Mb = np.max(log_odds)
					defaults['high_confidence_threshold'] = [mb,Mb]
				except:
					defaults['high_confidence_threshold'] = [0.,np.log(self.dm.g[-1]/(1.-self.dm.g[-1]))]
			else:
				defaults['high_confidence_threshold'] = [0.,np.log(self.dm.g[-1]/(1.-self.dm.g[-1]))]
		elif self.confidence_mapping_method=='belief':
			if invariant_decision_bounds:
				try:
					mb = min([2*np.min(self.dm.bounds[0])-1,1-2*np.min(self.dm.bounds[1])])-1e-3
					Mb = max([2*np.max(self.dm.bounds[0])-1,1-2*np.max(self.dm.bounds[1])])+1e-3
					defaults['high_confidence_threshold'] = [mb,Mb]
				except:
					defaults['high_confidence_threshold'] = [-0.001,1.001]
			else:
				defaults['high_confidence_threshold'] = [-0.001,1.001]
		else:
			raise ValueError('Unknown confidence_mapping_method: {0}'.format(self.confidence_mapping_method))
		defaults['confidence_map_slope'] = [0.,100.]
		if default_sp['high_confidence_threshold']>defaults['high_confidence_threshold'][1]:
			defaults['high_confidence_threshold'][1] = 2*default_sp['high_confidence_threshold']
		return defaults
	
	def default_optimizer_kwargs(self):
		"""
		self.default_optimizer_kwargs()
		
		Returns the default optimizer_kwargs that depend on the optimizer
		attribute
		
		"""
		if self.optimizer=='cma':
			return {'restarts':1,'restart_from_best':'False'}
		elif self.optimizer=='basinhopping':
			return {'stepsize':0.25, 'minimizer_kwargs':{'method':'Nelder-Mead'},'T':10.,'niter':100,'interval':10}
		elif self.optimizer in ['Brent','Bounded','Golden']: # Scalar minimize
			return {'disp': False, 'maxiter': 1000, 'repetitions': 10}
		else: # Multivariate minimize
			return {'disp': False, 'maxiter': 1000, 'maxfev': 10000, 'repetitions': 10}
	
	# Main fit method
	def fit(self,fixed_parameters={},start_point={},bounds={},optimizer_kwargs={},fit_arguments=None):
		"""
		self.fit(fixed_parameters={},start_point={},bounds={},optimizer_kwargs={},fit_arguments=None)
		
		Main Fitter function that executes the optimization procedure
		specified by the Fitter instance's optimizer attribute.
		This methods sets the fixed_parameters, start_point, bounds and
		optimizer_kwargs using the ones supplied in the input or using
		another Fitter instance's fit_arguments attribute.
		This method also sanitizes the fixed_parameters depending on
		the selected fitting method, init's the minimizer and returns
		the sanitized minimization output.
		
		For the detailed output form refer to the method sanitize_fmin_output
		
		"""
		if fit_arguments is None:
			self.set_fixed_parameters(fixed_parameters)
			self.set_start_point(start_point)
			self.set_bounds(bounds)
			self.fixed_parameters,self.fitted_parameters,self.start_point,self.bounds = self.sanitize_parameters_x0_bounds()
			
			self.set_optimizer_kwargs(optimizer_kwargs)
			self._fit_arguments = {'fixed_parameters':self.fixed_parameters,'fitted_parameters':self.fitted_parameters,\
								   'start_point':self.start_point,'bounds':self.bounds,'optimizer_kwargs':self.optimizer_kwargs}
		else:
			self.set_fit_arguments(fit_arguments)
		
		minimizer = self.init_minimizer(self.start_point,self.bounds,self.optimizer_kwargs)
		if self._is_binary_confidence:
			if self.method=='full':
				merit_function = self.full_merit
			elif self.method in ['binary_confidence_only','confidence_only']:
				merit_function = self.binary_confidence_only_merit
			elif self.method in ['full_binary_confidence','full_confidence']:
				merit_function = self.full_binary_confidence_merit
			else:
				raise ValueError('Unknown method {0}'.format(self.method))
		else:
			if self.method=='full':
				merit_function = self.full_merit
			elif self.method=='confidence_only':
				merit_function = self.confidence_only_merit
			elif self.method=='full_confidence':
				merit_function = self.full_confidence_merit
			elif self.method=='binary_confidence_only':
				merit_function = self.binary_confidence_only_merit
			elif self.method=='full_binary_confidence':
				merit_function = self.full_binary_confidence_merit
			else:
				raise ValueError('Unknown method {0}'.format(self.method))
		self.__fit_internals__ = None
		self._fit_output = minimizer(merit_function)
		self.__fit_internals__ = None
		return self._fit_output
	
	# Savers
	def save(self):
		"""
		self.save()
		
		Dumps the Fitter instances state to a pkl file.
		The Fitter's state is return by method self.__getstate__().
		The used pkl's file name is given by self.get_save_file_name().
		
		"""
		self.logger.debug('Fitter state that will be saved = "%s"',self.__getstate__())
		if not hasattr(self,'_fit_output'):
			raise ValueError('The Fitter instance has not performed any fit and still has no _fit_output attribute set')
		self.logger.info('Saving Fitter state to file "%s"',self.get_save_file_name())
		f = open(self.get_save_file_name(),'w')
		pickle.dump(self,f,pickle.HIGHEST_PROTOCOL)
		f.close()
	
	# Sanitizers
	def sanitize_parameters_x0_bounds(self):
		"""
		fitter.sanitize_parameters_x0_bounds()
		
		Some of the methods used to compute the merit assume certain
		parameters are fixed while others assume they are not. Furthermore
		some merit functions do not use all the parameters.
		This function allows the users to keep the flexibility of defining
		a single set of fixed parameters without worrying about the
		method specificities. The function takes the fixed_parameters,
		start_point and bounds specified by the user, and arranges them
		correctly for the specified merit method.
		
		Output:
		fixed_parameters,fitted_parameters,sanitized_start_point,sanitized_bounds
		
		fixed_parameters: A dict of parameter names as keys and their fixed values
		fitted_parameters: A list of the fitted parameters
		sanitized_start_point: A numpy ndarray with the fitted_parameters
		                       starting point
		sanitized_bounds: The fitted parameter's bounds. The specific
		                  format depends on the optimizer.
		
		"""
		_fixed_parameters = self._fixed_parameters.copy()
		for par in _fixed_parameters.keys():
			if _fixed_parameters[par] is None:
				_fixed_parameters[par] = self._start_point[par]
		
		binary_fixed_parameters = _fixed_parameters.copy()
		binary_fixed_parameters['confidence_map_slope'] = np.inf
		
		fittable_parameters = self.get_fittable_parameters()
		confidence_parameters = self.get_confidence_parameters()
		
		# Method specific fitted_parameters, fixed_parameters, starting points and bounds
		method_fitted_parameters = {'full_confidence':[],'full':[],'confidence_only':[],'binary_confidence_only':[],'full_binary_confidence':[]}
		method_fixed_parameters = {'full_confidence':_fixed_parameters.copy(),'full':_fixed_parameters.copy(),'confidence_only':_fixed_parameters.copy(),'binary_confidence_only':binary_fixed_parameters.copy(),'full_binary_confidence':binary_fixed_parameters.copy()}
		method_sp = {'full_confidence':[],'full':[],'confidence_only':[],'binary_confidence_only':[],'full_binary_confidence':[]}
		method_b = {'full_confidence':[],'full':[],'confidence_only':[],'binary_confidence_only':[],'full_binary_confidence':[]}
		for par in self._fitted_parameters:
			if par not in confidence_parameters:
				method_fitted_parameters['full'].append(par)
				method_sp['full'].append(self._start_point[par])
				method_b['full'].append(self._bounds[par])
				if par not in method_fixed_parameters['confidence_only'].keys():
					method_fixed_parameters['confidence_only'][par] = self._start_point[par]
				if par not in method_fixed_parameters['binary_confidence_only'].keys():
					method_fixed_parameters['binary_confidence_only'][par] = self._start_point[par]
				method_fitted_parameters['full_binary_confidence'].append(par)
				method_sp['full_binary_confidence'].append(self._start_point[par])
				method_b['full_binary_confidence'].append(self._bounds[par])
			else:
				method_fitted_parameters['confidence_only'].append(par)
				method_sp['confidence_only'].append(self._start_point[par])
				method_b['confidence_only'].append(self._bounds[par])
				if par not in method_fixed_parameters['full'].keys():
					method_fixed_parameters['full'][par] = self._start_point[par]
				if par!='confidence_map_slope':
					method_fitted_parameters['binary_confidence_only'].append(par)
					method_sp['binary_confidence_only'].append(self._start_point[par])
					method_b['binary_confidence_only'].append(self._bounds[par])
					method_fitted_parameters['full_binary_confidence'].append(par)
					method_sp['full_binary_confidence'].append(self._start_point[par])
					method_b['full_binary_confidence'].append(self._bounds[par])
			method_fitted_parameters['full_confidence'].append(par)
			method_sp['full_confidence'].append(self._start_point[par])
			method_b['full_confidence'].append(self._bounds[par])
		
		sanitized_start_point = np.array(method_sp[self.method])
		sanitized_bounds = list(np.array(method_b[self.method]).T)
		fitted_parameters = method_fitted_parameters[self.method]
		fixed_parameters = method_fixed_parameters[self.method]
		if len(fitted_parameters)==1 and self.optimizer=='cma':
			warnings.warn('CMA is unsuited for optimization of single dimensional parameter spaces. Optimizer was changed to Nelder-Mead')
			self.optimizer = 'Nelder-Mead'
		elif len(fitted_parameters)>1 and self.optimizer in ['Brent','Bounded','Golden']:
			raise ValueError('Brent, Bounded and Golden optimizers are only available for scalar functions. However, {0} parameters are being fitted. Please review the optimizer'.format(len(fitted_parameters)))
		
		if not (self.optimizer=='cma' or self.optimizer=='basinhopping'):
			sanitized_bounds = [(lb,ub) for lb,ub in zip(sanitized_bounds[0],sanitized_bounds[1])]
		self.logger.debug('Sanitized fixed parameters = %s',fixed_parameters)
		self.logger.debug('Sanitized fitted parameters = %s',fitted_parameters)
		self.logger.debug('Sanitized start_point = %s',sanitized_start_point)
		self.logger.debug('Sanitized bounds = %s',sanitized_bounds)
		return (fixed_parameters,fitted_parameters,sanitized_start_point,sanitized_bounds)
	
	def sanitize_fmin_output(self,output,package='cma'):
		"""
		self.sanitize_fmin_output(output,package='cma')
		
		The cma package returns the fit output in one format while the
		scipy package returns it in a completely different way.
		This method returns the fit output in a common format:
		It returns a tuple out
		
		out[0]: A dictionary with the fitted parameter names as keys and
		        the values being the best fitting parameter value.
		out[1]: Merit function value
		out[2]: Number of function evaluations
		out[3]: Overall number of function evaluations (in the cma
		        package, these can be more if there is noise handling)
		out[4]: Number of iterations
		out[5]: Mean of the sample of solutions
		out[6]: Std of the sample of solutions
		
		"""
		self.logger.debug('Sanitizing minizer output with package: {0}'.format(package))
		self.logger.debug('Output to sanitize: {0}'.format(output))
		if package=='cma':
			fitted_x = {}
			for index,par in enumerate(self.fitted_parameters):
				fitted_x[par] = output[0][index]
				if fitted_x[par]<self._bounds[par][0]:
					fitted_x[par] = self._bounds[par][0]
				elif fitted_x[par]>self._bounds[par][1]:
					fitted_x[par] = self._bounds[par][1]
			return (fitted_x,)+output[1:7]
		elif package=='scipy':
			fitted_x = {}
			for index,par in enumerate(self.fitted_parameters):
				fitted_x[par] = output.x[index]
				if fitted_x[par]<self._bounds[par][0]:
					fitted_x[par] = self._bounds[par][0]
				elif fitted_x[par]>self._bounds[par][1]:
					fitted_x[par] = self._bounds[par][1]
			return (fitted_x,output.fun,output.nfev,output.nfev,output.nit,output.x,np.nan*np.ones_like(output.x))
		elif package=='repeat_minimize':
			fitted_x = {}
			for index,par in enumerate(self.fitted_parameters):
				fitted_x[par] = output['xbest'][index]
				if fitted_x[par]<self._bounds[par][0]:
					fitted_x[par] = self._bounds[par][0]
				elif fitted_x[par]>self._bounds[par][1]:
					fitted_x[par] = self._bounds[par][1]
			return (fitted_x,output['funbest'],output['nfev'],output['nfev'],output['nit'],output['xmean'],output['xstd'])
		else:
			raise ValueError('Unknown package used for optimization. Unable to sanitize the fmin output')
	
	# Minimizer related methods
	def init_minimizer(self,start_point,bounds,optimizer_kwargs):
		"""
		self.init_minimizer(start_point,bounds,optimizer_kwargs)
		
		This method returns a callable to the minimization procedure.
		Said callable takes a single input argument, the minimization
		objective function, and returns a tuple with the sanitized
		minimization output. For more details on the output of the callable
		refer to the method sanitize_fmin_output
		
		Input:
		start_point: An array with the start point for the fitted parameters
		bounds: An array of shape (2,len(start_point)) that holds the
		        lower and upper bounds for each fitted parameter. Please
		        note that some of scipy's optimization methods ignore
		        the parameter bounds.
		optimizer_kwargs: A dict of options passed to each optimizer.
		                  Refer to the script's help for details.
		
		"""
		self.logger.debug('init_minimizer args: start_point=%(start_point)s, bounds=%(bounds)s, optimizer_kwargs=%(optimizer_kwargs)s',{'start_point':start_point,'bounds':bounds,'optimizer_kwargs':optimizer_kwargs})
		if self.optimizer=='cma':
			scaling_factor = bounds[1]-bounds[0]
			self.logger.debug('scaling_factor = %s',scaling_factor)
			options = {'bounds':bounds,'CMA_stds':scaling_factor,'verbose':1 if optimizer_kwargs['disp'] else -1}
			options.update(optimizer_kwargs)
			restarts = options['restarts']
			del options['restarts']
			restart_from_best = options['restart_from_best']
			del options['restart_from_best']
			del options['disp']
			options = cma.CMAOptions(options)
			minimizer = lambda x: self.sanitize_fmin_output(cma.fmin(x,start_point,1./3.,options,restarts=restarts,restart_from_best=restart_from_best),package='cma')
			#~ minimizer = lambda x: self.sanitize_fmin_output((start_point,None,None,None,None,None,None,None),'cma')
		elif self.optimizer=='basinhopping':
			options = optimizer_kwargs.copy()
			for k in options.keys():
				if k not in ['niter','T','stepsize','minimizer_kwargs','take_step','accept_test','callback','interval','disp','niter_success']:
					del options[k]
				elif k in ['take_step', 'accept_test', 'callback']:
					options[k] = eval(options[k])
			if 'take_step' not in options.keys():
				class Step_taker:
					def __init__(self,stepsize=options['stepsize']):
						self.stepsize = stepsize
						self.scaling_factor = bounds[1]-bounds[0]
					def __call__(self,x):
						x+= np.random.randn(*x.shape)*self.scaling_factor*self.stepsize
						return x
				options['take_step'] = Step_taker()
			if 'accept_test' not in options.keys():
				class Test_accepter:
					def __init__(self,bounds=bounds):
						self.bounds = bounds
					def __call__(self,**kwargs):
						return bool(np.all(np.logical_and(kwargs["x_new"]>=self.bounds[0],kwargs["x_new"]<=self.bounds[1])))
				options['accept_test'] = Test_accepter()
			if options['minimizer_kwargs']['method'] in ['CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'SLSQP', 'dogleg', 'trust-ncg']:
				jac_dx = self.get_jacobian_dx()
				epsilon = []
				for par in self.fitted_parameters:
					epsilon.append(jac_dx[par])
				def aux_function(f):
					options['minimizer_kwargs']['jac'] = lambda x: scipy.optimize.approx_fprime(x, f, epsilon)
					return self.sanitize_fmin_output(scipy.optimize.basinhopping(f, start_point, **options),package='scipy')
				minimizer = aux_function
			else:
				minimizer = lambda x: self.sanitize_fmin_output(scipy.optimize.basinhopping(x, start_point, **options),package='scipy')
		else:
			repetitions = optimizer_kwargs['repetitions']
			_start_points = [start_point]
			for rsp in np.random.rand(repetitions-1,len(start_point)):
				temp = []
				for val,(lb,ub) in zip(rsp,bounds):
					temp.append(val*(ub-lb)+lb)
				_start_points.append(np.array(temp))
			self.logger.debug('Array of start_points = {0}',_start_points)
			start_point_generator = iter(_start_points)
			if self.optimizer in ['CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'SLSQP', 'dogleg', 'trust-ncg']:
				jac_dx = self.get_jacobian_dx()
				epsilon = []
				for par in self.fitted_parameters:
					epsilon.append(jac_dx[par])
				jac = lambda x,f: scipy.optimize.approx_fprime(x, f, epsilon)
				minimizer = lambda f: self.sanitize_fmin_output(self.repeat_minimize(f,start_point_generator,bounds=bounds,optimizer_kwargs=optimizer_kwargs,jac=lambda x:jac(x,f)),package='repeat_minimize')
			else:
				minimizer = lambda f: self.sanitize_fmin_output(self.repeat_minimize(f,start_point_generator,bounds=bounds,optimizer_kwargs=optimizer_kwargs),package='repeat_minimize')
		return minimizer
	
	def repeat_minimize(self,merit,start_point_generator,bounds,optimizer_kwargs,jac=None):
		"""
		self.repeat_minimize(merit,start_point_generator,bounds,optimizer_kwargs,jac=None)
		
		A wrapper to repeat various rounds of minimization with the
		scipy.optimize.minimize or minimize_scalar method specified.
		
		Input:
			merit: The objective function used for the fits
			start_point_generator: A generator or iterable with the
				starting points that should be used for each fitting
				round
			bounds: The sanitized parameter bounds
			optimizer_kwargs: Additional kwargs to pass to the variable
				options of scipy.optimize.minimize or minimize_scalar
			jac: Can be None or a callable that computes the jacobian
				of the merit function
		
		Output:
			A dictionary with keys:
			xbest: Best solution
			funbest: Best solution function value
			nfev: Total number of function evaluations
			nit: Total number of iterations
			xs: The list of solutions for each round
			funs: The list of solutions' function values for each round
			xmean: The mean of the solutions across rounds
			xstd: The std of the solutions across rounds
			funmean: The mean of the solutions' function values across rounds
			funstd: The std of the solutions' function values across rounds
		
		"""
		output = {'xs':[],'funs':[],'nfev':0,'nit':0,'xbest':None,'funbest':None,'xmean':None,'xstd':None,'funmean':None,'funstd':None}
		repetitions = 0
		for start_point in start_point_generator:
			repetitions+=1
			self.logger.debug('Round {2} with start_point={0} and bounds={1}'.format(start_point, bounds,repetitions))
			if self.optimizer in ['Brent','Bounded','Golden']:
				res = scipy.optimize.minimize_scalar(merit,start_point,method=self.optimizer,\
								bounds=bounds[0], options=optimizer_kwargs)
			else:
				res = scipy.optimize.minimize(merit,start_point, method=self.optimizer,bounds=bounds,\
								options=optimizer_kwargs,jac=jac)
			self.logger.debug('New round with start_point={0} and bounds={0}'.format(start_point, bounds))
			self.logger.debug('Round {0} ended. Fun val: {1}. x={2}'.format(repetitions,res.fun,res.x))
			self.logger.debug('OptimizeResult: {0}'.format(res))
			try:
				nit = res.rit
			except:
				nit = 1
			if isinstance(res.x,float):
				x = np.array([res.x])
			else:
				x = res.x
			output['xs'].append(x)
			output['funs'].append(res.fun)
			output['nfev']+=res.nfev
			output['nit']+=nit
			if output['funbest'] is None or res.fun<output['funbest']:
				output['funbest'] = res.fun
				output['xbest'] = x
			self.logger.debug('Best so far: {0} at point {1}'.format(output['funbest'],output['xbest']))
		arr_xs = np.array(output['xs'])
		arr_funs = np.array(output['funs'])
		output['xmean'] = np.mean(arr_xs)
		output['xstd'] = np.std(arr_xs)
		output['funmean'] = np.mean(arr_funs)
		output['funstd'] = np.std(arr_funs)
		return output
	
	# Auxiliary method
	def confidence_mapping(self,parameters):
		"""
		self.high_confidence_mapping(parameters)
		
		Get the high confidence mapping as a function of time.
		Returns a numpy array of shape (2,self.dm.nT)
		The output[0] is the mapping for hits and output[1] is the
		mapping for misses.
		
		"""
		return self.dm.confidence_mapping(parameters['high_confidence_threshold'],parameters['confidence_map_slope'],self.confidence_mapping_method)
	
	# Method dependent merits
	def full_merit(self,x):
		"""
		self.full_merit(x)
		
		Returns the dataset's negative log likelihood (nLL) of jointly
		observing a given response time and performance for the supplied
		array of parameters x.
		
		Input:
			x: A numpy array that is converted to the parameter dict with
				a call to self.get_parameters_dict_from_array(x)
		
		Output:
			the nLL as a floating point
		
		"""
		parameters = self.get_parameters_dict_from_array(x)
		nlog_likelihood = 0.
		if 'cost' in self.get_fitted_parameters() or 'internal_var' in self.get_fitted_parameters():
			self.dm.set_cost(parameters['cost'])
			self.dm.set_internal_var(parameters['internal_var'])
			must_compute_first_passage_time = True
			must_store_first_passage_time = False
			xub,xlb = self.dm.xbounds()
		else:
			if self.__fit_internals__ is None:
				must_compute_first_passage_time = True
				must_store_first_passage_time = True
				self.dm.set_cost(parameters['cost'])
				self.dm.set_internal_var(parameters['internal_var'])
				xub,xlb = self.dm.xbounds()
				self.__fit_internals__ = {'xub':xub,'xlb':xlb,'first_passage_times':{}}
			else:
				must_compute_first_passage_time = False
				first_passage_times = self.__fit_internals__['first_passage_times']
		random_rt_likelihood = 0.5*parameters['phase_out_prob']/(self.max_RT-self.min_RT)
		if 'dead_time' in self.get_fitted_parameters() or 'dead_time_sigma' in self.get_fitted_parameters():
			dead_time_convolver = self.get_dead_time_convolver(parameters)
		else:
			if self.__fit_internals__ is None:
				dead_time_convolver = self.get_dead_time_convolver(parameters)
				self.__fit_internals__ = {'dead_time_convolver':dead_time_convolver}
			else:
				try:
					dead_time_convolver = self.__fit_internals__['dead_time_convolver']
				except:
					dead_time_convolver = self.get_dead_time_convolver(parameters)
					self.__fit_internals__['dead_time_convolver'] = dead_time_convolver
		if self.dm.known_variance():
			for index,drift in enumerate(self.mu):
				if must_compute_first_passage_time:
					first_passage_time = np.array(self.dm.fpt(drift,bounds=(xub,xlb)))
					if must_store_first_passage_time:
						self.__fit_internals__['first_passage_times'][drift] = first_passage_time
				else:
					first_passage_time = self.__fit_internals__['first_passage_times'][drift]
				gs = self.dm.rt_pdf(first_passage_time,dead_time_convolver=dead_time_convolver)
				t = np.arange(0,gs.shape[-1])*self.dm.dt
				indeces = self.mu_indeces==index
				for rt,perf in zip(self.rt[indeces],self.performance[indeces]):
					nlog_likelihood-= np.log(rt_likelihood(t,gs[1-int(perf)],rt)*(1-parameters['phase_out_prob'])+random_rt_likelihood)
		else:
			for index,stim in enumerate(self.unique_stim):
				drift = stim[0]
				external_var = stim[1]
				total_var = external_var + parameters['internal_var']
				if must_compute_first_passage_time:
					first_passage_time = np.array(self.dm.fpt(drift,total_var,bounds=(xub,xlb)))
					if must_store_first_passage_time:
						try:
							self.__fit_internals__['first_passage_times'][drift][external_var] = first_passage_time
						except:
							self.__fit_internals__['first_passage_times'][drift] = {}
							self.__fit_internals__['first_passage_times'][drift][external_var] = first_passage_time
				else:
					first_passage_time = self.__fit_internals__['first_passage_times'][drift][external_var]
				gs = self.dm.rt_pdf(first_passage_time,dead_time_convolver=dead_time_convolver)
				t = np.arange(0,gs.shape[-1])*self.dm.dt
				indeces = self.stim_indeces==index
				for rt,perf in zip(self.rt[indeces],self.performance[indeces]):
					nlog_likelihood-= np.log(rt_likelihood(t,gs[1-int(perf)],rt)*(1-parameters['phase_out_prob'])+random_rt_likelihood)
		return nlog_likelihood
	
	def confidence_only_merit(self,x):
		"""
		self.confidence_only_merit(x)
		
		Returns the dataset's negative log likelihood (nLL) of observing
		a given confidence for the supplied array of parameters x.
		
		Input:
			x: A numpy array that is converted to the parameter dict with
				a call to self.get_parameters_dict_from_array(x)
		
		Output:
			the nLL as a floating point
		
		"""
		parameters = self.get_parameters_dict_from_array(x)
		nlog_likelihood = 0.
		if self.__fit_internals__ is None:
			self.dm.set_cost(parameters['cost'])
			self.dm.set_internal_var(parameters['internal_var'])
			xub,xlb = self.dm.xbounds()
			must_compute_first_passage_time = True
			self.__fit_internals__ = {'xub':xub,'xlb':xlb,'first_passage_times':{}}
		else:
			must_compute_first_passage_time = False
			first_passage_times = self.__fit_internals__['first_passage_times']
		random_rt_likelihood = 0.5*parameters['phase_out_prob']/float(self.confidence_partition)
		mapped_confidences = self.confidence_mapping(parameters)
		if 'dead_time' in self.get_fitted_parameters() or 'dead_time_sigma' in self.get_fitted_parameters():
			dead_time_convolver = self.get_dead_time_convolver(parameters)
		else:
			if self.__fit_internals__ is None:
				dead_time_convolver = self.get_dead_time_convolver(parameters)
				self.__fit_internals__ = {'dead_time_convolver':dead_time_convolver}
			else:
				try:
					dead_time_convolver = self.__fit_internals__['dead_time_convolver']
				except:
					dead_time_convolver = self.get_dead_time_convolver(parameters)
					self.__fit_internals__['dead_time_convolver'] = dead_time_convolver
		
		if self.dm.known_variance():
			for index,drift in enumerate(self.mu):
				if must_compute_first_passage_time:
					gs = np.array(self.dm.fpt(drift,bounds=(xub,xlb)))
					self.__fit_internals__['first_passage_times'][drift] = gs
				else:
					gs = self.__fit_internals__['first_passage_times'][drift]
				conf_lik_pdf = np.sum(self.dm.rt_confidence_pdf(gs,mapped_confidences,dead_time_convolver),axis=2)
				indeces = self.mu_indeces==index
				for perf,conf in zip(self.performance[indeces],self.confidence[indeces]):
					nlog_likelihood-=np.log(confidence_likelihood(conf_lik_pdf[1-int(perf)],conf)*(1-parameters['phase_out_prob'])+random_rt_likelihood)
		else:
			for index,stim in enumerate(self.unique_stim):
				drift = stim[0]
				external_var = stim[1]
				total_var = external_var + parameters['internal_var']
				if must_compute_first_passage_time:
					gs = np.array(self.dm.fpt(drift,total_var,bounds=(xub,xlb)))
					try:
						self.__fit_internals__['first_passage_times'][drift][external_var] = gs
					except:
						self.__fit_internals__['first_passage_times'][drift] = {}
						self.__fit_internals__['first_passage_times'][drift][external_var] = gs
				else:
					gs = self.__fit_internals__['first_passage_times'][drift][external_var]
				conf_lik_pdf = np.sum(self.dm.rt_confidence_pdf(gs,confidence_response=mapped_confidences,dead_time_convolver=dead_time_convolver,confidence_partition=self.confidence_partition),axis=2)
				indeces = self.stim_indeces==index
				for perf,conf in zip(self.performance[indeces],self.confidence[indeces]):
					nlog_likelihood-=np.log(confidence_likelihood(conf_lik_pdf[1-int(perf)],conf)*(1-parameters['phase_out_prob'])+random_rt_likelihood)
		return nlog_likelihood
	
	def full_confidence_merit(self,x):
		"""
		self.confidence_only_merit(x)
		
		Returns the dataset's negative log likelihood (nLL) of jointly
		observing a given response time, confidence and performance for
		the supplied array of parameters x.
		
		Input:
			x: A numpy array that is converted to the parameter dict with
				a call to self.get_parameters_dict_from_array(x)
		
		Output:
			the nLL as a floating point
		
		"""
		parameters = self.get_parameters_dict_from_array(x)
		nlog_likelihood = 0.
		if 'cost' in self.get_fitted_parameters() or 'internal_var' in self.get_fitted_parameters():
			self.dm.set_cost(parameters['cost'])
			self.dm.set_internal_var(parameters['internal_var'])
			must_compute_first_passage_time = True
			must_store_first_passage_time = False
			xub,xlb = self.dm.xbounds()
		else:
			if self.__fit_internals__ is None:
				must_compute_first_passage_time = True
				must_store_first_passage_time = True
				self.dm.set_cost(parameters['cost'])
				self.dm.set_internal_var(parameters['internal_var'])
				xub,xlb = self.dm.xbounds()
				self.__fit_internals__ = {'xub':xub,'xlb':xlb,'first_passage_times':{}}
			else:
				must_compute_first_passage_time = False
				first_passage_times = self.__fit_internals__['first_passage_times']
		random_rt_likelihood = 0.5*parameters['phase_out_prob']/(self.max_RT-self.min_RT)/float(self.confidence_partition)
		mapped_confidences = self.confidence_mapping(parameters)
		if 'dead_time' in self.get_fitted_parameters() or 'dead_time_sigma' in self.get_fitted_parameters():
			dead_time_convolver = self.get_dead_time_convolver(parameters)
		else:
			if self.__fit_internals__ is None:
				dead_time_convolver = self.get_dead_time_convolver(parameters)
				self.__fit_internals__ = {'dead_time_convolver':dead_time_convolver}
			else:
				try:
					dead_time_convolver = self.__fit_internals__['dead_time_convolver']
				except:
					dead_time_convolver = self.get_dead_time_convolver(parameters)
					self.__fit_internals__['dead_time_convolver'] = dead_time_convolver
		if self.dm.known_variance():
			for index,drift in enumerate(self.mu):
				if must_compute_first_passage_time:
					gs = np.array(self.dm.fpt(drift,bounds=(xub,xlb)))
					if must_store_first_passage_time:
						self.__fit_internals__['first_passage_times'][drift] = gs
				else:
					gs = self.__fit_internals__['first_passage_times'][drift]
				rt_conf_lik_matrix = self.dm.rt_confidence_pdf(gs,confidence_response=mapped_confidences,dead_time_convolver=dead_time_convolver,confidence_partition=self.confidence_partition)
				t = np.arange(0,rt_conf_lik_matrix.shape[-1])*self.dm.dt
				indeces = self.mu_indeces==index
				for rt,perf,conf in zip(self.rt[indeces],self.performance[indeces],self.confidence[indeces]):
					nlog_likelihood-=np.log(rt_confidence_likelihood(t,rt_conf_lik_matrix[1-int(perf)],rt,conf)*(1-parameters['phase_out_prob'])+random_rt_likelihood)
		else:
			for index,stim in enumerate(self.unique_stim):
				drift = stim[0]
				external_var = stim[1]
				total_var = external_var + parameters['internal_var']
				if must_compute_first_passage_time:
					gs = np.array(self.dm.fpt(drift,total_var,bounds=(xub,xlb)))
					if must_store_first_passage_time:
						try:
							self.__fit_internals__['first_passage_times'][drift][external_var] = gs
						except:
							self.__fit_internals__['first_passage_times'][drift] = {}
							self.__fit_internals__['first_passage_times'][drift][external_var] = gs
				else:
					gs = self.__fit_internals__['first_passage_times'][drift][external_var]
				rt_conf_lik_matrix = self.dm.rt_confidence_pdf(gs,confidence_response=mapped_confidences,dead_time_convolver=dead_time_convolver,confidence_partition=self.confidence_partition)
				t = np.arange(0,rt_conf_lik_matrix.shape[-1])*self.dm.dt
				indeces = self.stim_indeces==index
				for rt,perf,conf in zip(self.rt[indeces],self.performance[indeces],self.confidence[indeces]):
					nlog_likelihood-=np.log(rt_confidence_likelihood(t,rt_conf_lik_matrix[1-int(perf)],rt,conf)*(1-parameters['phase_out_prob'])+random_rt_likelihood)
		return nlog_likelihood
	
	def full_binary_confidence_merit(self,x):
		"""
		self.confidence_only_merit(x)
		
		Returns the negative log likelihood (nLL) of jointly observing a given
		response time, performance and confidence below or above the
		median for the supplied array of parameters 'x'.
		
		Input:
			x: A numpy array that is converted to the parameter dict with
				a call to self.get_parameters_dict_from_array(x)
		
		Output:
			the nLL as a floating point
		
		"""
		parameters = self.get_parameters_dict_from_array(x)
		nlog_likelihood = 0.
		if 'cost' in self.get_fitted_parameters() or 'internal_var' in self.get_fitted_parameters():
			self.dm.set_cost(parameters['cost'])
			self.dm.set_internal_var(parameters['internal_var'])
			must_compute_first_passage_time = True
			must_store_first_passage_time = False
			xub,xlb = self.dm.xbounds()
		else:
			if self.__fit_internals__ is None:
				must_compute_first_passage_time = True
				must_store_first_passage_time = True
				self.dm.set_cost(parameters['cost'])
				self.dm.set_internal_var(parameters['internal_var'])
				xub,xlb = self.dm.xbounds()
				self.__fit_internals__ = {'xub':xub,'xlb':xlb,'first_passage_times':{}}
			else:
				must_compute_first_passage_time = False
				first_passage_times = self.__fit_internals__['first_passage_times']
		random_rt_likelihood = 0.25*parameters['phase_out_prob']/(self.max_RT-self.min_RT)
		mapped_confidences = self.dm.confidence_mapping(parameters['high_confidence_threshold'],np.inf,confidence_mapping_method=self.confidence_mapping_method)
		if 'dead_time' in self.get_fitted_parameters() or 'dead_time_sigma' in self.get_fitted_parameters():
			dead_time_convolver = self.get_dead_time_convolver(parameters)
		else:
			if self.__fit_internals__ is None:
				dead_time_convolver = self.get_dead_time_convolver(parameters)
				self.__fit_internals__ = {'dead_time_convolver':dead_time_convolver}
			else:
				try:
					dead_time_convolver = self.__fit_internals__['dead_time_convolver']
				except:
					dead_time_convolver = self.get_dead_time_convolver(parameters)
					self.__fit_internals__['dead_time_convolver'] = dead_time_convolver
		try:
			binary_confidence = self.__fit_internals__['binary_confidence']
		except:
			binary_confidence = self.get_binary_confidence_reports()
			if self.__fit_internals__ is None:
				self.__fit_internals__ = {'binary_confidence':binary_confidence}
			else:
				self.__fit_internals__['binary_confidence'] = binary_confidence
		
		if self.dm.known_variance():
			for index,drift in enumerate(self.mu):
				if must_compute_first_passage_time:
					gs = np.array(self.dm.fpt(drift,bounds=(xub,xlb)))
					if must_store_first_passage_time:
						self.__fit_internals__['first_passage_times'][drift] = gs
				else:
					gs = self.__fit_internals__['first_passage_times'][drift]
				binary_confidence_rt_pdf = self.dm.binary_confidence_rt_pdf(gs,mapped_confidences,dead_time_convolver=dead_time_convolver)
				t = np.arange(0,binary_confidence_rt_pdf.shape[-1])*self.dm.dt
				indeces = self.mu_indeces==index
				for rt,perf,binary_conf in zip(self.rt[indeces],self.performance[indeces],binary_confidence[indeces]):
					nlog_likelihood-=np.log(rt_likelihood(t,binary_confidence_rt_pdf[(1-int(perf)),binary_conf],rt)*(1-parameters['phase_out_prob'])+random_rt_likelihood)
		else:
			for index,stim in enumerate(self.unique_stim):
				drift = stim[0]
				external_var = stim[1]
				total_var = external_var + parameters['internal_var']
				if must_compute_first_passage_time:
					gs = np.array(self.dm.fpt(drift,total_var,bounds=(xub,xlb)))
					if must_store_first_passage_time:
						try:
							self.__fit_internals__['first_passage_times'][drift][external_var] = gs
						except:
							self.__fit_internals__['first_passage_times'][drift] = {}
							self.__fit_internals__['first_passage_times'][drift][external_var] = gs
				else:
					gs = self.__fit_internals__['first_passage_times'][drift][external_var]
				binary_confidence_rt_pdf = self.dm.binary_confidence_rt_pdf(gs,mapped_confidences,dead_time_convolver=dead_time_convolver)
				t = np.arange(0,binary_confidence_rt_pdf.shape[-1])*self.dm.dt
				indeces = self.stim_indeces==index
				for rt,perf,binary_conf in zip(self.rt[indeces],self.performance[indeces],binary_confidence[indeces]):
					nlog_likelihood-=np.log(rt_likelihood(t,binary_confidence_rt_pdf[(1-int(perf)),binary_conf],rt)*(1-parameters['phase_out_prob'])+random_rt_likelihood)
		return nlog_likelihood
	
	def binary_confidence_only_merit(self,x):
		"""
		self.confidence_only_merit(x)
		
		Returns the negative log likelihood (nLL) of jointly observing a given
		response time and confidence below or above the median for the
		supplied array of parameters 'x'.
		
		Input:
			x: A numpy array that is converted to the parameter dict with
				a call to self.get_parameters_dict_from_array(x)
		
		Output:
			the nLL as a floating point
		
		"""
		parameters = self.get_parameters_dict_from_array(x)
		nlog_likelihood = 0.
		if self.__fit_internals__ is None:
			self.dm.set_cost(parameters['cost'])
			self.dm.set_internal_var(parameters['internal_var'])
			xub,xlb = self.dm.xbounds()
			must_compute_first_passage_time = True
			self.__fit_internals__ = {'xub':xub,'xlb':xlb,'first_passage_times':{},'dead_time_convolver': self.get_dead_time_convolver(parameters)}
		else:
			must_compute_first_passage_time = False
			first_passage_times = self.__fit_internals__['first_passage_times']
		random_rt_likelihood = 0.5*parameters['phase_out_prob']
		mapped_confidences = self.dm.confidence_mapping(parameters['high_confidence_threshold'],np.inf,confidence_mapping_method=self.confidence_mapping_method)
		if 'dead_time' in self.get_fitted_parameters() or 'dead_time_sigma' in self.get_fitted_parameters():
			dead_time_convolver = self.get_dead_time_convolver(parameters)
		else:
			if self.__fit_internals__ is None:
				dead_time_convolver = self.get_dead_time_convolver(parameters)
				self.__fit_internals__ = {'dead_time_convolver':dead_time_convolver}
			else:
				try:
					dead_time_convolver = self.__fit_internals__['dead_time_convolver']
				except:
					dead_time_convolver = self.get_dead_time_convolver(parameters)
					self.__fit_internals__['dead_time_convolver'] = dead_time_convolver
		try:
			binary_confidence = self.__fit_internals__['binary_confidence']
		except:
			binary_confidence = self.get_binary_confidence_reports()
			if self.__fit_internals__ is None:
				self.__fit_internals__ = {'binary_confidence':binary_confidence}
			else:
				self.__fit_internals__['binary_confidence'] = binary_confidence
		
		if self.dm.known_variance():
			for index,drift in enumerate(self.mu):
				if must_compute_first_passage_time:
					gs = np.array(self.dm.fpt(drift,bounds=(xub,xlb)))
					self.__fit_internals__['first_passage_times'][drift] = gs
				else:
					gs = self.__fit_internals__['first_passage_times'][drift]
				binary_confidence_rt_pdf = np.sum(self.dm.binary_confidence_rt_pdf(gs,mapped_confidences,dead_time_convolver=dead_time_convolver),axis=0)
				t = np.arange(0,binary_confidence_rt_pdf.shape[-1])*self.dm.dt
				indeces = self.mu_indeces==index
				for rt,binary_conf in zip(self.rt[indeces],binary_confidence[indeces]):
					nlog_likelihood-=np.log(rt_likelihood(t,binary_confidence_rt_pdf[binary_conf],rt)*(1-parameters['phase_out_prob'])+random_rt_likelihood)
		else:
			for index,stim in enumerate(self.unique_stim):
				drift = stim[0]
				external_var = stim[1]
				total_var = external_var + parameters['internal_var']
				if must_compute_first_passage_time:
					gs = np.array(self.dm.fpt(drift,total_var,bounds=(xub,xlb)))
					try:
						self.__fit_internals__['first_passage_times'][drift][external_var] = gs
					except:
						self.__fit_internals__['first_passage_times'][drift] = {}
						self.__fit_internals__['first_passage_times'][drift][external_var] = gs
				else:
					gs = self.__fit_internals__['first_passage_times'][drift][external_var]
				binary_confidence_rt_pdf = np.sum(self.dm.binary_confidence_rt_pdf(gs,mapped_confidences,dead_time_convolver=dead_time_convolver),axis=0)
				t = np.arange(0,binary_confidence_rt_pdf.shape[-1])*self.dm.dt
				indeces = self.stim_indeces==index
				for rt,binary_conf in zip(self.rt[indeces],binary_confidence[indeces]):
					nlog_likelihood-=np.log(rt_likelihood(t,binary_confidence_rt_pdf[binary_conf],rt)*(1-parameters['phase_out_prob'])+random_rt_likelihood)
		return nlog_likelihood
	
	# Force to compute merit functions on an arbitrary parameter dict
	def forced_compute_full_merit(self,parameters):
		"""
		self.forced_compute_full_merit(parameters)
		
		The same as self.full_merit but on a parameter dict instead of a
		parameter array.
		
		"""
		nlog_likelihood = 0.
		self.dm.set_cost(parameters['cost'])
		self.dm.set_internal_var(parameters['internal_var'])
		xub,xlb = self.dm.xbounds()
		random_rt_likelihood = 0.5*parameters['phase_out_prob']/(self.max_RT-self.min_RT)
		dead_time_convolver = self.get_dead_time_convolver(parameters)
		if self.dm.known_variance():
			for index,drift in enumerate(self.mu):
				gs = self.dm.rt_pdf(np.array(self.dm.fpt(drift,bounds=(xub,xlb))),parameters,dead_time_convolver=dead_time_convolver)
				t = np.arange(0,gs.shape[-1])*self.dm.dt
				indeces = self.mu_indeces==index
				for rt,perf in zip(self.rt[indeces],self.performance[indeces]):
					nlog_likelihood-= np.log(rt_likelihood(t,gs[1-int(perf)],rt)*(1-parameters['phase_out_prob'])+random_rt_likelihood)
		else:
			for index,stim in enumerate(self.unique_stim):
				drift = stim[0]
				external_var = stim[1]
				total_var = external_var + parameters['internal_var']
				gs = self.dm.rt_pdf(np.array(self.dm.fpt(drift,total_var,bounds=(xub,xlb))),dead_time_convolver=dead_time_convolver)
				t = np.arange(0,gs.shape[-1])*self.dm.dt
				indeces = self.stim_indeces==index
				for rt,perf in zip(self.rt[indeces],self.performance[indeces]):
					nlog_likelihood-= np.log(rt_likelihood(t,gs[1-int(perf)],rt)*(1-parameters['phase_out_prob'])+random_rt_likelihood)
		return nlog_likelihood
	
	def forced_compute_confidence_only_merit(self,parameters):
		"""
		self.forced_compute_confidence_only_merit(parameters)
		
		The same as self.confidence_only_merit but on a parameter dict
		instead of a parameter array.
		
		"""
		nlog_likelihood = 0.
		self.dm.set_cost(parameters['cost'])
		self.dm.set_internal_var(parameters['internal_var'])
		xub,xlb = self.dm.xbounds()
		random_rt_likelihood = 0.5*parameters['phase_out_prob']/float(self.confidence_partition)
		mapped_confidences = self.confidence_mapping(parameters)
		dead_time_convolver = self.get_dead_time_convolver(parameters)
		
		if self.dm.known_variance():
			for index,drift in enumerate(self.mu):
				gs = np.array(self.dm.fpt(drift,bounds=(xub,xlb)))
				conf_lik_pdf = np.sum(self.dm.rt_confidence_pdf(gs,confidence_response=mapped_confidences,dead_time_convolver=dead_time_convolver,confidence_partition=self.confidence_partition),axis=2)
				indeces = self.mu_indeces==index
				for perf,conf in zip(self.performance[indeces],self.confidence[indeces]):
					nlog_likelihood-=np.log(confidence_likelihood(conf_lik_pdf[1-int(perf)],conf)*(1-parameters['phase_out_prob'])+random_rt_likelihood)
		else:
			for index,stim in enumerate(self.unique_stim):
				drift = stim[0]
				external_var = stim[1]
				total_var = external_var + parameters['internal_var']
				gs = np.array(self.dm.fpt(drift,total_var,bounds=(xub,xlb)))
				conf_lik_pdf = np.sum(self.dm.rt_confidence_pdf(gs,confidence_response=mapped_confidences,dead_time_convolver=dead_time_convolver,confidence_partition=self.confidence_partition),axis=2)
				indeces = self.stim_indeces==index
				for perf,conf in zip(self.performance[indeces],self.confidence[indeces]):
					nlog_likelihood-=np.log(confidence_likelihood(conf_lik_pdf[1-int(perf)],conf)*(1-parameters['phase_out_prob'])+random_rt_likelihood)
		return nlog_likelihood
	
	def forced_compute_full_confidence_merit(self,parameters):
		"""
		self.forced_compute_full_confidence_merit(parameters)
		
		The same as self.full_confidence_merit but on a parameter dict
		instead of a parameter array.
		
		"""
		nlog_likelihood = 0.
		self.dm.set_cost(parameters['cost'])
		self.dm.set_internal_var(parameters['internal_var'])
		xub,xlb = self.dm.xbounds()
		random_rt_likelihood = 0.5*parameters['phase_out_prob']/(self.max_RT-self.min_RT)/float(self.confidence_partition)
		mapped_confidences = self.confidence_mapping(parameters)
		dead_time_convolver = self.get_dead_time_convolver(parameters)
		
		if self.dm.known_variance():
			for index,drift in enumerate(self.mu):
				gs = np.array(self.dm.fpt(drift,bounds=(xub,xlb)))
				rt_conf_lik_matrix = self.dm.rt_confidence_pdf(gs,confidence_response=mapped_confidences,dead_time_convolver=dead_time_convolver,confidence_partition=self.confidence_partition)
				t = np.arange(0,rt_conf_lik_matrix.shape[-1])*self.dm.dt
				indeces = self.mu_indeces==index
				for rt,perf,conf in zip(self.rt[indeces],self.performance[indeces],self.confidence[indeces]):
					nlog_likelihood-=np.log(rt_confidence_likelihood(t,rt_conf_lik_matrix[1-int(perf)],rt,conf)*(1-parameters['phase_out_prob'])+random_rt_likelihood)
		else:
			for index,stim in enumerate(self.unique_stim):
				drift = stim[0]
				external_var = stim[1]
				total_var = external_var + parameters['internal_var']
				gs = np.array(self.dm.fpt(drift,total_var,bounds=(xub,xlb)))
				rt_conf_lik_matrix = self.dm.rt_confidence_pdf(gs,confidence_response=mapped_confidences,dead_time_convolver=dead_time_convolver,confidence_partition=self.confidence_partition)
				t = np.arange(0,rt_conf_lik_matrix.shape[-1])*self.dm.dt
				indeces = self.stim_indeces==index
				for rt,perf,conf in zip(self.rt[indeces],self.performance[indeces],self.confidence[indeces]):
					nlog_likelihood-=np.log(rt_confidence_likelihood(t,rt_conf_lik_matrix[1-int(perf)],rt,conf)*(1-parameters['phase_out_prob'])+random_rt_likelihood)
		return nlog_likelihood
	
	def forced_compute_full_binary_confidence_merit(self,parameters):
		"""
		self.forced_compute_full_binary_confidence_merit(parameters)
		
		The same as self.full_binary_confidence_merit but on a parameter dict
		instead of a parameter array.
		
		"""
		nlog_likelihood = 0.
		self.dm.set_cost(parameters['cost'])
		self.dm.set_internal_var(parameters['internal_var'])
		xub,xlb = self.dm.xbounds()
		random_rt_likelihood = 0.25*parameters['phase_out_prob']/(self.max_RT-self.min_RT)
		mapped_confidences = self.dm.confidence_mapping(parameters['high_confidence_threshold'],np.inf,confidence_mapping_method=self.confidence_mapping_method)
		dead_time_convolver = self.get_dead_time_convolver(parameters)
		binary_confidence = self.get_binary_confidence_reports()
		
		if self.dm.known_variance():
			for index,drift in enumerate(self.mu):
				gs = np.array(self.dm.fpt(drift,bounds=(xub,xlb)))
				binary_confidence_rt_pdf = self.dm.binary_confidence_rt_pdf(gs,mapped_confidences,dead_time_convolver=dead_time_convolver)
				t = np.arange(0,binary_confidence_rt_pdf.shape[-1])*self.dm.dt
				indeces = self.mu_indeces==index
				for rt,perf,binary_conf in zip(self.rt[indeces],self.performance[indeces],binary_confidence[indeces]):
					nlog_likelihood-=np.log(rt_likelihood(t,binary_confidence_rt_pdf[(1-int(perf)),binary_conf],rt)*(1-parameters['phase_out_prob'])+random_rt_likelihood)
		else:
			for index,stim in enumerate(self.unique_stim):
				drift = stim[0]
				external_var = stim[1]
				total_var = external_var + parameters['internal_var']
				gs = np.array(self.dm.fpt(drift,total_var,bounds=(xub,xlb)))
				binary_confidence_rt_pdf = self.dm.binary_confidence_rt_pdf(gs,mapped_confidences,dead_time_convolver=dead_time_convolver)
				t = np.arange(0,binary_confidence_rt_pdf.shape[-1])*self.dm.dt
				indeces = self.stim_indeces==index
				for rt,perf,binary_conf in zip(self.rt[indeces],self.performance[indeces],binary_confidence[indeces]):
					nlog_likelihood-=np.log(rt_likelihood(t,binary_confidence_rt_pdf[(1-int(perf)),binary_conf],rt)*(1-parameters['phase_out_prob'])+random_rt_likelihood)
		return nlog_likelihood
	
	def forced_compute_binary_confidence_only_merit(self,parameters):
		"""
		self.forced_compute_binary_confidence_only_merit(parameters)
		
		The same as self.binary_confidence_only_merit but on a parameter dict
		instead of a parameter array.
		
		"""
		nlog_likelihood = 0.
		self.dm.set_cost(parameters['cost'])
		self.dm.set_internal_var(parameters['internal_var'])
		xub,xlb = self.dm.xbounds()
		dead_time_convolver = self.get_dead_time_convolver(parameters)
		random_rt_likelihood = 0.5*parameters['phase_out_prob']
		mapped_confidences = self.dm.confidence_mapping(parameters['high_confidence_threshold'],np.inf,confidence_mapping_method=self.confidence_mapping_method)
		binary_confidence = self.get_binary_confidence_reports()
		
		if self.dm.known_variance():
			for index,drift in enumerate(self.mu):
				gs = np.array(self.dm.fpt(drift,bounds=(xub,xlb)))
				binary_confidence_rt_pdf = np.sum(self.dm.binary_confidence_rt_pdf(gs,mapped_confidences,dead_time_convolver=dead_time_convolver),axis=0)
				t = np.arange(0,binary_confidence_rt_pdf.shape[-1])*self.dm.dt
				indeces = self.mu_indeces==index
				for rt,binary_conf in zip(self.rt[indeces],binary_confidence[indeces]):
					nlog_likelihood-=np.log(rt_likelihood(t,binary_confidence_rt_pdf[binary_conf],rt)*(1-parameters['phase_out_prob'])+random_rt_likelihood)
		else:
			for index,stim in enumerate(self.unique_stim):
				drift = stim[0]
				external_var = stim[1]
				total_var = external_var + parameters['internal_var']
				gs = np.array(self.dm.fpt(drift,total_var,bounds=(xub,xlb)))
				binary_confidence_rt_pdf = np.sum(self.dm.binary_confidence_rt_pdf(gs,mapped_confidences,dead_time_convolver=dead_time_convolver),axis=0)
				t = np.arange(0,binary_confidence_rt_pdf.shape[-1])*self.dm.dt
				indeces = self.stim_indeces==index
				for rt,binary_conf in zip(self.rt[indeces],binary_confidence[indeces]):
					nlog_likelihood-=np.log(rt_likelihood(t,binary_confidence_rt_pdf[binary_conf],rt)*(1-parameters['phase_out_prob'])+random_rt_likelihood)
		return nlog_likelihood
	
	# Theoretical predictions
	def theoretical_rt_confidence_distribution(self,fit_output=None,binary_confidence=None):
		"""
		self.theoretical_rt_confidence_distribution(fit_output=None,binary_confidence=None)
		
		Returns the theoretically predicted joint probability density of
		RT, performance and confidence.
		
		Input:
			fit_output: If None, it uses the instances fit_output. It is
				used to extract the model parameters for the computation
			binary_confidence: None or a Bool used to indicate whether to
				force binary confidence pdf or not. If None, it returns
				binary or not depending on the Fitter's method attribute
		
		Output:
			(pdf,t)
			pdf: A numpy array that can be like the output from a call
				to self.rt_confidence_pdf(...) or
				self.binary_confidence_rt_pdf(...) depending on the
				binary_confidence input.
			t: The time array over which the pdf is computed
		
		"""
		if binary_confidence is None:
			binary_confidence = self.method=='binary_confidence_only' or self.method=='full_binary_confidence'
		parameters = self.get_parameters_dict_from_fit_output(fit_output)
		
		self.dm.set_cost(parameters['cost'])
		self.dm.set_internal_var(parameters['internal_var'])
		xub,xlb = self.dm.xbounds()
		
		if binary_confidence:
			mapped_confidences = self.dm.confidence_mapping(parameters['high_confidence_threshold'],np.inf,confidence_mapping_method=self.confidence_mapping_method)
		else:
			mapped_confidences = self.dm.confidence_mapping(parameters['high_confidence_threshold'],parameters['confidence_map_slope'],confidence_mapping_method=self.confidence_mapping_method)
		dead_time_convolver = self.get_dead_time_convolver(parameters)
		output = None
		if self.dm.known_variance():
			for index,drift in enumerate(self.mu):
				gs = np.array(self.dm.fpt(drift,bounds=(xub,xlb)))
				if binary_confidence:
					rt_conf_lik_matrix = self.dm.binary_confidence_rt_pdf(gs,mapped_confidences,dead_time_convolver)
				else:
					rt_conf_lik_matrix = self.dm.rt_confidence_pdf(gs,mapped_confidences,dead_time_convolver)
				
				if output is None:
					output = rt_conf_lik_matrix*self.mu_prob[index]
				else:
					output+= rt_conf_lik_matrix*self.mu_prob[index]
		else:
			for index,stim in enumerate(self.unique_stim):
				drift = stim[0]
				total_var = stim[1] + parameters['internal_var']
				stim_prob = self.stim_prob[index]
				gs = np.array(self.dm.fpt(drift,total_var,bounds=(xub,xlb)))
				if binary_confidence:
					rt_conf_lik_matrix = self.dm.binary_confidence_rt_pdf(gs,mapped_confidences,dead_time_convolver)
				else:
					rt_conf_lik_matrix = self.dm.rt_confidence_pdf(gs,mapped_confidences,dead_time_convolver)
				if output is None:
					output = rt_conf_lik_matrix*stim_prob
				else:
					output+= rt_conf_lik_matrix*stim_prob
		output/=(np.sum(output)*self.dm.dt)
		t = np.arange(0,output.shape[2],dtype=np.float)*self.dm.dt
		random_rt_likelihood = np.ones_like(output)
		random_rt_likelihood[:,:,np.logical_or(t<self.min_RT,t>self.max_RT)] = 0.
		random_rt_likelihood/=(np.sum(random_rt_likelihood)*self.dm.dt)
		return output*(1.-parameters['phase_out_prob'])+parameters['phase_out_prob']*random_rt_likelihood, t
	
	# Plotter
	def plot_fit(self,fit_output=None,saver=None,show=True,is_binary_confidence=None):
		"""
		self.plot_fit(fit_output=None,saver=None,show=True)
		
		This function actually only performs the following
		self.get_fitter_plot_handler(fit_output=fit_output).plot(saver=saver,show=show,is_binary_confidence=is_binary_confidence)
		
		The first part only gets the corresponding Fitter_plot_handler
		instance and then calls the handler's plot method. For a detailed
		description of the plot's functionallity refer to
		Fitter_plot_handler.plot
		
		"""
		if not can_plot:
			raise ImportError('Could not import matplotlib package and it is imposible to plot fit')
		
		self.get_fitter_plot_handler(fit_output=fit_output).plot(saver=saver,show=show,is_binary_confidence=is_binary_confidence)
	
	def stats(self,fit_output=None,binary_confidence=None,return_mean_rt=False,
				return_mean_confidence=False,return_median_rt=False,
				return_median_confidence=False,return_std_rt=False,return_std_confidence=False,
				return_auc=False):
		"""
		self.stats(fit_output=None,binary_confidence=None,return_mean_rt=False,
				return_mean_confidence=False,return_median_rt=False,
				return_median_confidence=False,return_std_rt=False,return_std_confidence=False,
				return_auc=False)
		
		This function computes some relevant statistics predicted by the
		fitted model. The fitted parameters are used if the input
		fit_output is None, and if it is not None it uses the provided
		parameters.
		The input binary_confidence can be provided to override the
		default confidence behavior. If None, the default confidence
		behavior is used (continuous or binary). If it is True, then
		the computations force binary confidence. This means that all
		the conditioned statistics over confidence will be for low (index
		0) and high (index 1) confidence values. If it is False, then
		the computations force continuous confidence values. This means
		that all the conditioned statistics over confidence will be
		computed on the array numpy.linspace(0,1,self.confidence_partition).
		
		It can compute the following:
		performance: A float with the model's predicted overall performance.
		performance_conditioned: A numpy array of floats with the model's
			performance conditioned to a given confidence report. The
			confidence values are numpy.linspace(0,1,self.confidence_partition)
			if the model does not use binary confidence. If the model
			uses binary confidence, the confidence values are [0,1].
			The default binary confidence can be overriden with the
			input binary_confidence.
		mean_rt: A float with the mean overall response time. Is only
			returned if return_mean_rt is True.
		mean_rt_perf: A 1D numpy array of floats with the mean response
			time conditioned to performance. Is only returned if
			return_mean_rt is True.
		mean_rt_conditioned: A 2D numpy array of floats with the mean
			response time conditioned to performance (first axis) and
			confidence values (second axis). Is only returned if
			return_mean_rt is True.
		median_rt: A float with the median overall response time. Is only
			returned if return_median_rt is True.
		median_rt_perf: A 1D numpy array of floats with the median
			response time conditioned to performance. Is only returned
			if return_mean_rt is True.
		median_rt_conf: A 1D numpy array of floats with the median
			response time conditioned to confidence. Is only returned
			if return_mean_rt is True.
		median_rt_conditioned: A 2D numpy array of floats with the median
			response time conditioned to performance (first axis) and
			confidence values (second axis). Is only returned if
			return_median_rt is True.
		std_rt: A float with the std overall response time. Is only
			returned if return_std_rt is True.
		std_rt_conditioned: A 2D numpy array of floats with the std
			response time conditioned to performance (first axis) and
			confidence values (second axis). Is only returned if
			return_std_rt is True.
		mean_confidence: A float with the mean overall confidence. Is
			only returned if return_mean_confidence is True.
		mean_confidence_conditioned: A numpy array of floats with the
			mean confidence conditioned to performance. Is only returned
			if return_mean_confidence is True.
		median_confidence: A float with the median overall confidence.
			Is only returned if return_median_confidence is True.
		median_confidence_conditioned: A numpy array of floats with the
			median confidence conditioned to performance. Is only
			returned if return_median_confidence is True.
		std_confidence: A float with the std overall confidence. Is only
			returned if return_std_confidence is True.
		std_confidence_conditioned: A numpy array of floats with the std
			confidence conditioned to performance. Is only returned if
			return_std_confidence is True.
		auc: A float with the area under the Receiver Operating
			Characteristic (ROC) curve. Is only returned if return_auc
			is True.
		
		Output:
			A dictionary with keys equal to the above mentioned statistic
			names.
		
		IMPORTANT NOTE:
		The 'Auditivo' task forces the subjects to listen to stimuli
		and to respond after the stimulation has ended. Our model does
		not have any method to pause the decision until the stimulation
		dissapears and, in fact, the model actually forms the decision
		during the stimulation. In the 'Auditivo' experimental data, the
		subject's rt are measured relative to the stimulation's end,
		while our model's rt is measured relative to the stimulation's
		start. To make the raw experimental data stats compatible with
		the Fitter stats, we substract the _forced_non_decision_time to
		all of the time related stats.
		This issue does not affect any of the plotting methods because
		the Fitter internally stores the subject's rt with the added
		_forced_non_decision_time
		
		"""
		pdf,t = self.theoretical_rt_confidence_distribution(fit_output=fit_output)
		binary_confidence = self.method=='binary_confidence_only' or self.method=='full_binary_confidence'
		dt = self.dm.dt
		valid = t<=self._time_available_to_respond
		if not all(valid):
			t = t[valid]
			pdf = pdf[:,:,valid]
			pdf/=(np.sum(pdf)*dt)
		# Some tasks force the subjects to perceive the stimuli and
		# respond after the stimulation has ended. Our model does not
		# have any method to pause the decision until the stimulation
		# dissapears and, in fact, the model actually forms the decision
		# during the stimulation. Furthermore, in some experiments the
		# response time is measured relative to the stimulation's end,
		# while our model's rt is measured relative to the stimulation's
		# start. To make the raw experimental data stats compatible with
		# the Fitter stats, we substract the forced_non_decision_time
		# to all of the time related stats.
		if self._rt_measured_from_stim_end:
			t-=self._forced_non_decision_time
		c = np.linspace(0,1,pdf.shape[1])
		dc = c[1]-c[0]
		performance = np.sum(pdf[0]*dt)
		performance_conditioned = np.sum(pdf,axis=2)*dt
		if return_mean_rt or return_std_rt:
			any_perf_conf_pdf = np.sum(np.sum(pdf,axis=0),axis=0)
			mean_rt = np.sum(any_perf_conf_pdf*t*dt)
			mean_rt_perf = np.sum((pdf*t).reshape((pdf.shape[0],-1)),axis=1)/np.sum(pdf.reshape((pdf.shape[0],-1)),axis=1)
			mean_rt_conditioned = np.sum(pdf*t,axis=2)/np.sum(pdf,axis=2)
			if return_std_rt:
				std_rt = np.sqrt(np.sum(any_perf_conf_pdf*(t-mean_rt)**2*dt))
				std_rt_conditioned = np.sqrt(np.sum(pdf*(t[None,None,:]-mean_rt_conditioned[:,:,None])**2,axis=2)/np.sum(pdf,axis=2))
		if return_median_rt:
			any_perf_conf_pdf = np.sum(np.sum(pdf,axis=0),axis=0)
			median_rt = np.interp(0.5,np.cumsum(any_perf_conf_pdf)*dt,t)
			cumpdf = np.cumsum(pdf,axis=2)/np.sum(pdf,axis=2,keepdims=True)
			cumpdf_perf = np.cumsum(np.sum(pdf,axis=1),axis=1)/np.sum(np.sum(pdf,axis=1),axis=1,keepdims=True)
			cumpdf_conf = np.cumsum(np.sum(pdf,axis=0),axis=1)/np.sum(np.sum(pdf,axis=0),axis=1,keepdims=True)
			median_rt_perf = np.zeros(pdf.shape[0])
			median_rt_conf = np.zeros(pdf.shape[1])
			median_rt_conditioned = np.zeros(tuple(cumpdf.shape[:2]))
			for row in range(cumpdf.shape[0]):
				median_rt_perf[row] = np.interp(0.5,cumpdf_perf[row],t)
				for col in range(cumpdf.shape[1]):
					if row==0:
						median_rt_conf[col] = np.interp(0.5,cumpdf_conf[col],t)
					median_rt_conditioned[row,col] = np.interp(0.5,cumpdf[row,col],t)
		if return_mean_confidence or return_std_confidence:
			any_rt_pdf = np.sum(pdf,axis=2)*dt
			any_perf_rt_pdf = np.sum(any_rt_pdf,axis=0)
			mean_confidence = np.sum(any_perf_rt_pdf*c)
			mean_confidence_conditioned = np.sum(any_rt_pdf*c,axis=1)/np.sum(any_rt_pdf,axis=1)
			if return_std_confidence:
				std_confidence = np.sqrt(np.sum(any_perf_rt_pdf*(c-mean_confidence)**2))
				std_confidence_conditioned = np.sqrt(np.sum(any_rt_pdf*(c[None,:]-mean_confidence_conditioned[:,None])**2,axis=1)/np.sum(any_rt_pdf,axis=1))
		if return_median_confidence:
			any_rt_pdf = np.sum(pdf,axis=2)*dt
			any_perf_rt_pdf = np.sum(any_rt_pdf,axis=0)
			median_confidence = np.interp(0.5,np.cumsum(any_perf_rt_pdf),c)
			cumpdf = np.cumsum(any_rt_pdf,axis=1)/np.sum(any_rt_pdf,axis=1,keepdims=True)
			median_confidence_conditioned = np.zeros(cumpdf.shape[0])
			for row in range(cumpdf.shape[0]):
				median_confidence_conditioned[row] = np.interp(0.5,cumpdf[row],c)
		if return_auc:
			import scipy.integrate
			pconf_hit = np.hstack((np.zeros(1),np.cumsum(np.sum(pdf[0],axis=1)*dt)))
			pconf_hit/=pconf_hit[-1]
			pconf_miss = np.hstack((np.zeros(1),np.cumsum(np.sum(pdf[1],axis=1)*dt)))
			pconf_miss/=pconf_miss[-1]
			auc = scipy.integrate.trapz(pconf_miss,pconf_hit)
		output = {'performance':performance,'performance_conditioned':performance_conditioned}
		if return_mean_rt:
			output['mean_rt'] = mean_rt
			output['mean_rt_perf'] = mean_rt_perf
			output['mean_rt_conditioned'] = mean_rt_conditioned
		if return_mean_confidence:
			output['mean_confidence'] = mean_confidence
			output['mean_confidence_conditioned'] = mean_confidence_conditioned
		if return_median_rt:
			output['median_rt'] = median_rt
			output['median_rt_perf'] = median_rt_perf
			output['median_rt_conf'] = median_rt_conf
			output['median_rt_conditioned'] = median_rt_conditioned
		if return_median_confidence:
			output['median_confidence'] = median_confidence
			output['median_confidence_conditioned'] = median_confidence_conditioned
		if return_std_rt:
			output['std_rt'] = std_rt
			output['std_rt_conditioned'] = std_rt_conditioned
		if return_std_confidence:
			output['std_confidence'] = std_confidence
			output['std_confidence_conditioned'] = std_confidence_conditioned
		if return_auc:
			output['auc'] = auc
		return output

class Fitter_plot_handler():
	def __init__(self,obj,binary_split_method='median'):
		"""
		Fitter_plot_handler(dictionary,binary_split_method='median')
		
		This class implements a flexible way to handle data ploting for
		separate subjectSessions and their corresponding fitted models
		encoded by a Fitter class instance. However, the most important
		feature of the Fitter_plot_handler is that it is easy to
		merge several different subjectSessions data and Fitter
		theoretical predictions. This is done by holding an internal
		dictionary that holds the experimental and theoretical probability
		densities for response times, confidence, and joint RT-confidence
		distributions for hit and miss trials. It also holds the array
		of times over which the RT distribution were computed and the
		array of confidence values over which the confidence distribution
		were computed. This is done for keys that represent the experiment,
		session and subject name. These keys can be aliased and merged
		easily. Then the data can be plotted with the plot method.
		
		Input:
			dictionary: a dict with the following hierarchy
				{key:{'experimental':{'t_array':numpy.ndarray,
									  'c_array':numpy.ndarray,
									  'rt':numpy.ndarray,
									  'confidence':numpy.ndarray,
									  'hit_histogram':numpy.ndarray,
									  'miss_histogram':numpy.ndarray},
					  'theoretical':{'t_array':numpy.ndarray,
									  'c_array':numpy.ndarray,
									  'rt':numpy.ndarray,
									  'confidence':numpy.ndarray,
									  'hit_histogram':numpy.ndarray,
									  'miss_histogram':numpy.ndarray}}}
				The base key must have the following format
				{experiment}_subject_{name}_session_{session}.
				When merging the data of every subject or session, or
				both, the keys fields _subject_{name} and/or _session_{session}
				are removed and the inner 't_array', 'rt', 'confidence',
				'hit_histogram' and 'miss_histogram's are merged.
				
				The 'experimental' key holds the subjectSession's data.
				The 'theoretical' key holds the theoretical predictions.
				The 't_array' is the array of times on which the
				probability densities were computed (in the theoretical)
				or the edges of the histogram with which the 'experimental'
				'rt' probabilities were computed.
				The 'c_array' is the same as the 't_array' but for
				confidence.
				The 'rt' holds the response time histogram (experimental)
				or the probability density (theoretical).
				The 'confidence' holds the confidence histogram (experimental)
				or the  probability density (theoretical).
				The 'hit_histogram' and 'miss_histogram' are 2D numpy
				arrays that hold the 2D histograms (experimental)
				or probability densities (theoretical) of the confidence
				and response time. The first index corresponds to
				confidence and the second to time.
			'binary_split_method': Can be 'median', 'mean' or 'half'.
				This speficies the way in which the continuous confidence
				reports will be binarized by default. This can be
				overriden in the plot method. Refer to
				Fitter.get_binary_confidence for a detailed description
				of the three methods. [Default 'median']
		
		In order to merge to different Fitter_plot_handler instances 'a'
		and 'b', it is only necessary to do sum them. Assume that 'a' has
		a single key 'ka' and 'b' has a single key 'kb'
		
		c = a+b
		
		The c instance will have a single key if 'ka'=='kb' and two keys
		if 'ka'!='kb'. These keys can then be merged with the desired
		rule by doing:
		
		c = c.merge(merge='all')
		
		WARNING: When summing two Fitter_plot_handlers, the left instance's
		binary_split_method will be kept and the other instance's
		attribute value will be ignored.
		
		"""
		self.logger = logging.getLogger("fits_module.Fitter_plot_handler")
		self.binary_split_method = binary_split_method
		self.required_data = ['hit_histogram','miss_histogram','rt','confidence','t_array','c_array']
		# For backward compatibility
		new_style_required = ['t_array','c_array']
		self.categories = ['experimental','theoretical']
		self.dictionary = {}
		try:
			for key in obj.keys():
				self.dictionary[key] = {}
				for category in self.categories:
					self.dictionary[key][category] = {}
					if any([not req in obj[key][category] for req in new_style_required]):
						# Old style handler
						self.dictionary[key][category]['t_array'] = copy.deepcopy(obj[key][category]['hit_histogram']['x'])
						self.dictionary[key][category]['c_array'] = copy.deepcopy(obj[key][category]['hit_histogram']['y'])
						self.dictionary[key][category]['hit_histogram'] = copy.deepcopy(obj[key][category]['hit_histogram']['z'])
						self.dictionary[key][category]['miss_histogram'] = copy.deepcopy(obj[key][category]['miss_histogram']['z'])
						self.dictionary[key][category]['rt'] = copy.deepcopy(obj[key][category]['rt']['y'])
						self.dictionary[key][category]['confidence'] = copy.deepcopy(obj[key][category]['confidence']['y'])
					else:
						# New style handler
						for required_data in self.required_data:
							self.dictionary[key][category][required_data] = copy.deepcopy(obj[key][category][required_data])
		except:
			raise RuntimeError('Invalid object used to init Fitter_plot_handler')
	
	def keys(self):
		return self.dictionary.keys()
	
	def __getitem__(self,key):
		return self.dictionary[key]
	
	def __setitem__(self,key,value):
		self.dictionary[key] = value
	
	def __aliased_iadd__(self,other,key_aliaser=lambda key:key):
		for other_key in other.keys():
			if key_aliaser(other_key) in self.keys():
				for category in self.categories:
					torig = self[key_aliaser(other_key)][category]['t_array']
					tadded = other[other_key][category]['t_array']
					for required_data in ['hit_histogram','miss_histogram','rt']:
						orig = self[key_aliaser(other_key)][category][required_data]
						added = other[other_key][category][required_data]
						if len(torig)<len(tadded):
							summed = added+np.pad(orig, ((0,0),(0,len(tadded)-len(torig))),str('constant'),constant_values=(0., 0.))
							self[key_aliaser(other_key)][category][required_data] = summed
						elif len(torig)>len(tadded):
							summed = orig+np.pad(added, ((0,0),(0,len(torig)-len(tadded))),str('constant'),constant_values=(0., 0.))
							self[key_aliaser(other_key)][category][required_data] = summed
						else:
							self[key_aliaser(other_key)][category][required_data]+= added
					if len(torig)<len(tadded):
						self[key_aliaser(other_key)][category]['t_array'] = copy.copy(tadded)
					self[key_aliaser(other_key)][category]['confidence']+= \
							other[other_key][category]['confidence']
			else:
				self[key_aliaser(other_key)] = {}
				for category in self.categories:
					self[key_aliaser(other_key)][category] = {}
					for required_data in self.required_data:
						self[key_aliaser(other_key)][category][required_data] = \
							copy.deepcopy(other[other_key][category][required_data])
		return self
	
	def __iadd__(self,other):
		return self.__aliased_iadd__(other)
	
	def __add__(self,other):
		output = Fitter_plot_handler(self,self.binary_split_method)
		output+=other
		return output
	
	def normalize(self,in_place=True):
		"""
		self.normalize(in_place=True)
		
		For every key and 'theoretical' and 'experimental' inner
		categories, normalize the 'rt', 'confidence', 'hit_histogram' and
		'miss_histogram' distributions.
		
		If in_place is True, the normalization is done on the
		Fitter_plot_handler caller instance. Be aware that this means
		that it will not be safe to add it with other unnormalized
		Fitter_plot_handler instances. If in_place is False, it returns a
		new normalized Fitter_plot_handler instance.
		
		"""
		self.logger.debug('Normalizing Fitter_plot_handler in place? {0}'.format(in_place))
		if in_place:
			out = self
		else:
			out = Fitter_plot_handler(self,self.binary_split_method)
		for key in out.keys():
			for category in self.categories:
				dt = out[key][category]['t_array'][1]-out[key][category]['t_array'][0]
				out[key][category]['rt']/=(np.sum(out[key][category]['rt'])*dt)
				out[key][category]['confidence']/=np.sum(out[key][category]['confidence'])
				out[key][category]['hit_histogram']/=(np.sum(out[key][category]['hit_histogram'])*dt)
				out[key][category]['miss_histogram']/=(np.sum(out[key][category]['miss_histogram'])*dt)
		return out
	
	def plot(self,saver=None,show=True,xlim_rt_cutoff=True,fig=None,logscale=True,
			is_binary_confidence=None,binary_split_method=None):
		"""
		plot(self,saver=None,show=True,xlim_rt_cutoff=True,fig=None,logscale=True,
			is_binary_confidence=None,binary_split_method=None)
		
		Main plotting routine has two completely different output forms
		that depend on the parameter is_binary_confidence. If
		is_binary_confidence is True (or if it is None but the
		Fitter_plot_handler caller was constructed from a binary
		confidence Fitter instance) then this function produces a figure
		with 4 axes distributed as a subplot(22i).
		subplot(221) will hold the rt distribution of hits and misses
		subplot(222) will hold the rt distribution of high and low
		confidence reports
		subplot(223) will hold the rt distribution of high and low hit
		confidence reports
		subplot(224) will hold the rt distribution of high and low miss
		confidence reports
		
		If is_binary_confidence is False (or if it is None but the
		Fitter_plot_handler caller was constructed from a continuous
		confidence Fitter instance) then this function produces a figure
		with 6 axes distributed as a subplot(32i).
		subplot(321) will hold the rt distribution of hits and misses
		subplot(322) will hold the confidence distribution of hits and
		misses. This plot can be in logscale if the input logscale is
		True.
		subplot(323) and subplot(324) will hold the experimental joint
		rt-confidence distributions of hits and misses respectively.
		subplot(325) and subplot(326) will hold the theoretical joint
		rt-confidence distributions of hits and misses respectively.
		All four of the above mentioned graph are affected by the
		logscale input parameter. If True, the colorscale is logarithmic.
		
		Other input arguments:
		xlim_rt_cutoff: A bool that indicates whether to set the
			theoretical graphs xlim equal to experimental xlim for all
			plots that involve response times.
		show: A bool indicating whether to show the figure after it has
			been created and freezing the execution until it is closed.
		saver: If it is not None, saver must be a string or an object
			that implements the savefig method similar to
			matplotlib.pyplot.savefig. If it is a string it will be used
			to save the figure as:
			matplotlib.pyplot.savefig(saver,,bbox_inches='tight')
		binary_split_method: Override the way in which the continuous
			confidence reports are binarized. Available methods are
			None, 'median', 'half' and 'mean'. If None, the
			Fitter_plot_handler's binary_split_method attribute will be
			used. If supplied value is not None, the binarization method
			will be overriden. Be aware that this parameter will affect
			the subject's data and it will also affect the Fitter's data
			only if the Fitter's data is encoded in a continuous way.
			If said data is already binary, the binary_split_method will
			have no additional effect. These methods are only used
			when is_binary_confidence is True. For a detailed
			description of the three methods mentioned above, refer to
			Fitter.get_binary_confidence.
		
		"""
		if not can_plot:
			raise ImportError('Could not import matplotlib package and it is imposible to produce any plot')
		handler = self.normalize(in_place=False)
		
		if binary_split_method is None:
			binary_split_method = self.binary_split_method
		if binary_split_method is None:
			self.logger.debug('Fitter_plot_handler was not saved with the binary_split_method attribute. Will assume median split but this may have not been the split method used to generate the handler')
			binary_split_method = 'median'
		
		for key in sorted(handler.keys()):
			self.logger.info('Preparing to plot key {0}'.format(key))
			subj = handler.dictionary[key]['experimental']
			model = handler.dictionary[key]['theoretical']
			
			if is_binary_confidence is None:
				is_binary_confidence = len(model['c_array'])==2
			self.logger.debug('Is binary confidence? {0}'.format(is_binary_confidence))
			
			rt_cutoff = subj['t_array'][-1]+0.5*(subj['t_array'][-1]-subj['t_array'][-2])
			
			if fig is None:
				fig = plt.figure(figsize=(10,12))
				self.logger.debug('Created figure instance {0}'.format(fig.number))
			else:
				self.logger.debug('Will use figure instance {0}'.format(fig.number))
				fig.clf()
				plt.figure(fig.number)
				self.logger.debug('Cleared figure instance {0} and setted it as the current figure'.format(fig.number))
			if not is_binary_confidence:
				self.logger.debug('Starting the continuous confidence plotting procedure')
				gs1 = gridspec.GridSpec(1, 2,left=0.10, right=0.90, top=0.95,bottom=0.70)
				gs2 = gridspec.GridSpec(2, 2,left=0.10, right=0.85, wspace=0.05, hspace=0.05, top=0.62,bottom=0.05)
				gs3 = gridspec.GridSpec(1, 1,left=0.87, right=0.90, wspace=0.1, top=0.62,bottom=0.05)
				self.logger.debug('Created gridspecs')
				axrt = plt.subplot(gs1[0])
				self.logger.debug('Created rt axes')
				dt = subj['t_array'][1]-subj['t_array'][0]
				dc = subj['c_array'][1]-subj['c_array'][0]
				if len(subj['t_array'])==len(subj['rt'][0]):
					self.logger.debug('Subject "t_array" holds the histogram centers. Converting to edges to achieve proper step centering.')
					subj_t_array = np.hstack((subj['t_array']-0.5*dt,subj['t_array'][-1:]+0.5*dt))
					subj_c_array = np.hstack((subj['c_array']-0.5*dc,subj['c_array'][-1:]+0.5*dc))
				else:
					self.logger.debug('Subject "t_array" holds the histogram edges')
					subj_t_array = subj['t_array']
					subj_c_array = subj['c_array']
				t_extent = [subj['t_array'][0]-0.5*dt,subj['t_array'][-1]+0.5*dt]
				subj_rt = np.hstack((subj['rt'],np.array([subj['rt'][:,-1]]).T))
				subj_confidence = np.hstack((subj['confidence'],np.array([subj['confidence'][:,-1]]).T))
				plt.step(subj_t_array,subj_rt[0],'b',label='Subject hit',where='post')
				plt.step(subj_t_array,subj_rt[1],'r',label='Subject miss',where='post')
				plt.plot(model['t_array'],model['rt'][0],'b',label='Model hit',linewidth=3)
				plt.plot(model['t_array'],model['rt'][1],'r',label='Model miss',linewidth=3)
				self.logger.debug('Plotted rt axes')
				if xlim_rt_cutoff:
					axrt.set_xlim([0,rt_cutoff])
				plt.xlabel('RT [s]')
				plt.ylabel('Prob density')
				plt.legend(loc='best', fancybox=True, framealpha=0.5)
				self.logger.debug('Completed rt axes plot, legend and labels')
				axconf = plt.subplot(gs1[1])
				self.logger.debug('Created confidence axes')
				plt.step(subj_c_array,subj_confidence[0],'b',label='Subject hit',where='post')
				if model['confidence'].shape[1]==2:
					model['c_array'] = np.array([0,1])
				plt.plot(model['c_array'],model['confidence'][0],'b',label='Model hit',linewidth=3)
				if logscale:
					plt.step(subj_c_array,subj_confidence[1],'r',label='Subject miss',where='post')
					plt.plot(model['c_array'],model['confidence'][1],'r',label='Model miss',linewidth=3)
					axconf.set_yscale('log')
				else:
					#~ plt.step(subj['c_array'],-subj_confidence[1],'r',label='Subject miss')
					#~ plt.plot(model['c_array'],-model['confidence'][1],'r',label='Model miss',linewidth=3)
					plt.step(subj_c_array,subj_confidence[1],'r',label='Subject miss',where='post')
					plt.plot(model['c_array'],model['confidence'][1],'r',label='Model miss',linewidth=3)
				self.logger.debug('Plotted confidence axes')
				plt.xlabel('Confidence')
				plt.legend(loc='best', fancybox=True, framealpha=0.5)
				#~ gs1.tight_layout(fig,rect=(0.10, 0.70, 0.90, 0.95), pad=0, w_pad=0.03)
				self.logger.debug('Completed confidence axes plot, legend and labels')
				
				if logscale:
					vmin = np.min([np.min(subj['hit_histogram'][(subj['hit_histogram']>0).nonzero()]),
								   np.min(subj['miss_histogram'][(subj['miss_histogram']>0).nonzero()])])
					norm = LogNorm()
				else:
					vmin = np.min([np.min(subj['hit_histogram']),
								   np.min(subj['miss_histogram']),
								   np.min(model['hit_histogram']),
								   np.min(model['miss_histogram'])])
					norm = None
				vmax = np.max([np.max([subj['hit_histogram'],subj['miss_histogram']]),
							   np.max([model['hit_histogram'],model['miss_histogram']])])
				
				ax00 = plt.subplot(gs2[0,0])
				self.logger.debug('Created subject hit axes')
				plt.imshow(subj['hit_histogram'],aspect="auto",interpolation='none',origin='lower',vmin=vmin,vmax=vmax,
							extent=[t_extent[0],t_extent[1],0,1],norm=norm)
				plt.ylabel('Confidence')
				plt.title('Hit')
				self.logger.debug('Populated subject hit axes')
				ax10 = plt.subplot(gs2[1,0],sharex=ax00,sharey=ax00)
				self.logger.debug('Created model hit axes')
				plt.imshow(model['hit_histogram'],aspect="auto",interpolation='none',origin='lower',vmin=vmin,vmax=vmax,
							extent=[model['t_array'][0],model['t_array'][-1],0,1],norm=norm)
				plt.xlabel('RT [s]')
				plt.ylabel('Confidence')
				self.logger.debug('Populated model hit axes')
				
				ax01 = plt.subplot(gs2[0,1],sharex=ax00,sharey=ax00)
				self.logger.debug('Created subject miss axes')
				plt.imshow(subj['miss_histogram'],aspect="auto",interpolation='none',origin='lower',vmin=vmin,vmax=vmax,
							extent=[t_extent[0],t_extent[1],0,1],norm=norm)
				plt.title('Miss')
				self.logger.debug('Populated subject miss axes')
				ax11 = plt.subplot(gs2[1,1],sharex=ax00,sharey=ax00)
				self.logger.debug('Created model miss axes')
				im = plt.imshow(model['miss_histogram'],aspect="auto",interpolation='none',origin='lower',vmin=vmin,vmax=vmax,
							extent=[model['t_array'][0],model['t_array'][-1],0,1],norm=norm)
				plt.xlabel('RT [s]')
				if xlim_rt_cutoff:
					ax00.set_xlim([0,rt_cutoff])
				self.logger.debug('Populated model miss axes')
				
				ax00.tick_params(labelleft=True, labelbottom=False)
				ax01.tick_params(labelleft=False, labelbottom=False)
				ax10.tick_params(labelleft=True, labelbottom=True)
				ax11.tick_params(labelleft=False, labelbottom=True)
				self.logger.debug('Completed histogram axes')
				
				cbar_ax = plt.subplot(gs3[0])
				self.logger.debug('Created colorbar axes')
				plt.colorbar(im, cax=cbar_ax)
				plt.ylabel('Prob density')
				self.logger.debug('Completed colorbar axes')
				
				plt.suptitle(key)
				self.logger.debug('Sucessfully completed figure for key {0}'.format(key))
			else: # binary confidence
				self.logger.debug('Starting the binary confidence plotting procedure')
				gs1 = gridspec.GridSpec(2, 2,left=0.10, right=0.90, top=0.95,bottom=0.1)
				self.logger.debug('Created gridspecs')
				axrt = plt.subplot(gs1[0,0])
				self.logger.debug('Created rt axes')
				dt = subj['t_array'][1]-subj['t_array'][0]
				subj_performance = np.sum(subj['rt'][0])*dt
				self.logger.debug('Binary confidence split method: {0}'.format(binary_split_method))
				if binary_split_method=='median':
					subj_split_ind = (np.cumsum(np.sum(subj['confidence'],axis=0))>=0.5).nonzero()[0][0]
					if model['confidence'].shape[1]>2:
						self.logger.debug('Model confidence data is not natively binary. Binarizing now...')
						model_split_ind = (np.cumsum(np.sum(model['confidence'],axis=0))>=0.5).nonzero()[0][0]
					else:
						model_split_ind = 1
				elif binary_split_method=='half':
					subj_split_ind = (subj['c_array']>=0.5).nonzero()[0][0]
					if model['confidence'].shape[1]>2:
						self.logger.debug('Model confidence data is not natively binary. Binarizing now...')
						model_split_ind = (model['c_array']>=0.5).nonzero()[0][0]
					else:
						model_split_ind = 1
				elif binary_split_method=='mean':
					if len(subj['c_array'])>(subj['confidence'].shape[1]):
						c_array = np.array([0.5*(e1+e0) for e1,e0 in zip(subj['c_array'][1:],subj['c_array'][:-1])])
					else:
						c_array = subj['c_array']
					subj_split_ind = (c_array>=np.sum(subj['confidence']*c_array)).nonzero()[0][0]
					if model['confidence'].shape[1]>2:
						self.logger.debug('Model confidence data is not natively binary. Binarizing now...')
						model_split_ind = (model['c_array']>=np.sum(model['confidence']*model['c_array'])).nonzero()[0][0]
					else:
						model_split_ind = 1
				self.logger.debug('Subject confidence split index: {0}'.format(subj_split_ind))
				self.logger.debug('Model confidence split index: {0}'.format(model_split_ind))
				subj_lowconf_rt = np.array([np.sum(subj['hit_histogram'][:subj_split_ind],axis=0)*subj_performance,
											np.sum(subj['miss_histogram'][:subj_split_ind],axis=0)*(1-subj_performance)])
				subj_highconf_rt = np.array([np.sum(subj['hit_histogram'][subj_split_ind:],axis=0)*subj_performance,
											np.sum(subj['miss_histogram'][subj_split_ind:],axis=0)*(1-subj_performance)])
				model_performance = np.sum(model['rt'][0])*(model['t_array'][1]-model['t_array'][0])
				model_low_rt = np.array([np.sum(model['hit_histogram'][:model_split_ind],axis=0)*model_performance,
										np.sum(model['miss_histogram'][:model_split_ind],axis=0)*(1-model_performance)])
				model_high_rt = np.array([np.sum(model['hit_histogram'][model_split_ind:],axis=0)*model_performance,
										np.sum(model['miss_histogram'][model_split_ind:],axis=0)*(1-model_performance)])
				if len(subj['t_array'])==len(subj['rt'][0]):
					self.logger.debug('Subject "t_array" holds the histogram centers. Converting to edges to achieve proper step centering.')
					subj_t_array = np.hstack((subj['t_array']-0.5*dt,subj['t_array'][-1:]+0.5*dt))
				else:
					self.logger.debug('Subject "t_array" holds the histogram edges')
					subj_t_array = subj['t_array']
				subj_rt = np.hstack((subj['rt'],np.array([subj['rt'][:,-1]]).T))
				subj_lowconf_rt = np.hstack((subj_lowconf_rt,np.array([subj_lowconf_rt[:,-1]]).T))
				subj_highconf_rt = np.hstack((subj_highconf_rt,np.array([subj_highconf_rt[:,-1]]).T))
				
				plt.step(subj_t_array,subj_rt[0],'b',label='Subject hit',where='post')
				plt.step(subj_t_array,subj_rt[1],'r',label='Subject miss',where='post')
				plt.plot(model['t_array'],model['rt'][0],'b',label='Model hit',linewidth=3)
				plt.plot(model['t_array'],model['rt'][1],'r',label='Model miss',linewidth=3)
				self.logger.debug('Plotted rt axes')
				if xlim_rt_cutoff:
					axrt.set_xlim([0,rt_cutoff])
				plt.ylabel('Prob density')
				plt.legend(loc='best', fancybox=True, framealpha=0.5)
				self.logger.debug('Completed rt axes plot, legend and labels')
				
				axconf = plt.subplot(gs1[0,1])
				self.logger.debug('Created confidence axes')
				plt.step(subj_t_array,np.sum(subj_lowconf_rt,axis=0),'mediumpurple',label='Subject low',where='post')
				plt.step(subj_t_array,np.sum(subj_highconf_rt,axis=0),'forestgreen',label='Subject high',where='post')
				plt.plot(model['t_array'],np.sum(model_low_rt,axis=0),'mediumpurple',label='Model low',linewidth=3)
				plt.plot(model['t_array'],np.sum(model_high_rt,axis=0),'forestgreen',label='Model high',linewidth=3)
				if xlim_rt_cutoff:
					axconf.set_xlim([0,rt_cutoff])
				plt.legend(loc='best', fancybox=True, framealpha=0.5)
				self.logger.debug('Plotted confidence axes')
				
				axhitconf = plt.subplot(gs1[1,0])
				self.logger.debug('Created hit confidence axes')
				plt.step(subj_t_array,subj_lowconf_rt[0],'mediumpurple',label='Subject hit low',where='post')
				plt.step(subj_t_array,subj_highconf_rt[0],'forestgreen',label='Subject hit high',where='post')
				plt.plot(model['t_array'],model_low_rt[0],'mediumpurple',label='Model hit low',linewidth=3)
				plt.plot(model['t_array'],model_high_rt[0],'forestgreen',label='Model hit high',linewidth=3)
				plt.xlabel('RT [s]')
				plt.ylabel('Prob density')
				if xlim_rt_cutoff:
					axhitconf.set_xlim([0,rt_cutoff])
				plt.legend(loc='best', fancybox=True, framealpha=0.5)
				self.logger.debug('Plotted hit confidence axes')
				
				axmissconf = plt.subplot(gs1[1,1])
				self.logger.debug('Created miss confidence axes')
				plt.step(subj_t_array,subj_lowconf_rt[1],'mediumpurple',label='Subject miss low',where='post')
				plt.step(subj_t_array,subj_highconf_rt[1],'forestgreen',label='Subject miss high',where='post')
				plt.plot(model['t_array'],model_low_rt[1],'mediumpurple',label='Model miss low',linewidth=3)
				plt.plot(model['t_array'],model_high_rt[1],'forestgreen',label='Model miss high',linewidth=3)
				plt.xlabel('RT [s]')
				if xlim_rt_cutoff:
					axmissconf.set_xlim([0,rt_cutoff])
				plt.legend(loc='best', fancybox=True, framealpha=0.5)
				self.logger.debug('Plotted miss confidence axes')
				self.logger.debug('Completed confidence axes plot, legend and labels')
				
				plt.suptitle(key)
				self.logger.debug('Sucessfully completed figure for key {0}'.format(key))
			
			if saver:
				self.logger.debug('Saving figure')
				if isinstance(saver,str):
					plt.savefig(saver,bbox_inches='tight')
				else:
					saver.savefig(fig,bbox_inches='tight')
			if show:
				self.logger.debug('Showing figure')
				plt.show(True)
				fig = None
	
	def save(self,fname):
		self.logger.debug('Fitter_plot_handler state that will be saved = "%s"',self.__getstate__())
		self.logger.info('Saving Fitter_plot_handler state to file "%s"',fname)
		f = open(fname,'w')
		pickle.dump(self,f,pickle.HIGHEST_PROTOCOL)
		f.close()
	
	def __setstate__(self,state):
		self.logger = logging.getLogger("fits_module.Fitter_plot_handler")
		if 'binary_split_method' in state.keys():
			binary_split_method = state['binary_split_method']
		else:
			binary_split_method = None
			self.logger.debug('Old version of Fitter_plot_handler without the binary_split_method attribute. Will set it to None to explicitly set it appart from the default median split initialization.')
		self.__init__(state['dictionary'],binary_split_method)
	
	def __getstate__(self):
		return {'dictionary':self.dictionary,'binary_split_method':self.binary_split_method}
	
	def merge(self,merge='all'):
		"""
		self.merge(merge='all')
		
		Returns a new Fitter_plot_handler instance that merges the keys
		of the caller instance. The merge input can be 'all', 'subjects'
		or 'sessions'.
		If 'all', all the keys with the same experiment are merged together.
		If 'subjects', the different subjects are merged together yielding
		keys that only identify the experiment and session.
		If 'sessions', the different sessions are merged together yielding
		keys that only identify the experiment and subject name.
		
		"""
		if merge=='subjects':
			key_aliaser = lambda key: re.sub('_subject_[\[\]\-0-9]+','',key)
		elif merge=='sessions':
			key_aliaser = lambda key: re.sub('_session_[\[\]\-0-9]+','',key)
		elif merge=='all':
			key_aliaser = lambda key: re.sub('_session_[\[\]\-0-9]+','',re.sub('_subject_[\[\]\-0-9]+','',key))
		else:
			raise ValueError('Unknown merge option={0}'.format(merge))
		output = Fitter_plot_handler({},self.binary_split_method)
		return output.__aliased_iadd__(self,key_aliaser)

def parse_input():
	script_help = """ moving_bounds_fits.py help
 Sintax:
 fits_module.py [option flag] [option value]
 
 fits_module.py -h [or --help] displays help
 
 Optional arguments are:
 '-t' or '--task': Integer that identifies the task number when running multiple tasks
                   in parallel. By default it is one based but this behavior can be
                   changed with the option --task_base. [Default 1]
 '-nt' or '--ntasks': Integer that identifies the number tasks working in parallel [Default 1]
 '-tb' or '--task_base': Integer that identifies the task base. Can be 0 or 1, indicating
                         the task number of the root task. [Default 1]
 '-m' or '--method': String that identifies the fit method. Available values are full,
                     confidence_only, full_confidence, binary_confidence_only
                     and full_binary_confidence. [Default full]
 '-o' or '--optimizer': String that identifies the optimizer used for fitting.
                        Available values are 'cma', scipy's 'basinhopping',
                        all the scipy.optimize.minimize and 
                        scipy.optimizer.minimize_scalar methods.
                        WARNING, cma is suited for problems with more than one dimensional
                        parameter spaces. If the optimization is performed on a single
                        dimension, the optimizer is changed to 'Nelder-Mead' before
                        processing the supplied optimizer_kwargs.
                        If one of the minimize_scalar methods is supplied
                        but more than one parameter is being fitted, a ValueError
                        is raised. [Default cma]
 '-s' or '--save': This flag takes no values. If present it saves the figure.
 '--save_plot_handler': This flag takes no value. If present, the plot_handler is saved.
 '--load_plot_handler': This flag takes no value. If present, the plot_handler is loaded from the disk.
 '--save_stats': This flag takes no value. If present, the Fitter.stats() dictionary is saved.
 '--show': This flag takes no values. If present it displays the plotted figure
           and freezes execution until the figure is closed.
 '--fit': This flag takes no values. If present it performs the fit for the selected
          method. By default, this flag is always set.
 '--no-fit': This flag takes no values. If present no fit is performed for the selected
             method. This flag should be used when it is only necesary to plot the results.
 '-sf' or '--suffix': A string suffix to paste to the filenames. [Default '']
 '--rt_cutoff': A Float that specifies the maximum RT in seconds to accept when
                loading subject data. Note that the "Luminancia" experiment
                forced subjects to respond in less than 1 second. Hence,
                an rt_cutoff greater than 1 second is supplied for the
                "Luminancia" experiment, it is chopped down to 1 second.
                [Defaults to 1 second for the Luminancia experiment
                and 14 seconds for the other experiments]
 '--plot_handler_rt_cutoff': Same as rt_cutoff but is used to create the
                             subject's RT histogram when constructing
                             the Fitter_plot_handler. [Default the fitter's rt_cutoff.
                             IMPORTANT! Fitter instances are saved with the rt_cutoff
                             value they were created with. If a fitter instance was
                             loaded from a file in order to get the Fitter_plot_handler,
                             then its rt_cutoff may be different than the value
                             specified in a separate run of the script.]
 '--plot_binary': Can be None, True or False. It is used to override the
                  plotting handling of binary or continuous confidence.
                  If None, the confidence is handled automatically with
                  the natural binary or continuous confidence depending
                  on the supplied method. If True, it forces the binary
                  confidence plot. If False, it forces the continuous
                  confidence plot. [Default None]
 '--confidence_partition': An Int that specifies the number of bins in which to partition
                           the [0,1] confidence report interval [Default 100]
 '--merge': Can be None, 'all', 'all_sessions' or 'all_subjects'. This parameter
            controls if and how the subject-session data should be merged before
            performing the fits. If merge is set to 'all', all the data is merged
            into a single "subjectSession". If merge is 'all_sessions', the
            data across all sessions for the same subject is merged together.
            If merge is 'all_subjects', the data across all subjects for a
            single session is merged. For all the above, the experiments are
            always treated separately. If merge is None, the data of every
            subject and session is treated separately. [Default None]
 '--plot_merge': Can be None, 'all', 'sessions' or 'subjects'. This parameter
            controls if and how the subject-session data should be merged when
            performing the plots. If merge is set to 'all', all subjects
            and sessions data are pooled together to plot figures that
            correspond to each experiment. If set to 'sessions' the sessions
            are pooled together and a separate figure is created for each
            subject and experiment. If set to 'subject', the subjects are
            pooled together and separate figures for each experiment and
            session are created. WARNING! This option does not override
            the --merge option. If the --merge option was set, the data
            that is used to perform the fits is merged together and
            cannot be divided again. --plot_merge allows the user to fit
            the data for every subject, experiment and session independently
            but to pool them together while plotting. [Default None]
 '-e' or '--experiment': Can be 'all', 'luminancia', '2afc' or 'auditivo'.
                         Indicates the experiment that you wish to fit. If set to
                         'all', all experiment data will be fitted. [Default 'all']
                         WARNING: is case insensitive.
 '-g' or '--debug': Activates the debug messages
 '-v' or '--verbose': Activates info messages (by default only warnings and errors
                      are printed).
 '-w': Override an existing saved fitter. This flag is only used if the 
       '--fit' flag is not disabled. If the flag '-w' is supplied, the script
       will override the saved fitter instance associated to the fitted
       subjectSession. If this flag is not supplied, the script will skip
       the subjectSession's that were already fitted.
 '-hcm' or '--confidence_mapping_method': Select the method with
       which the confidence mapping is computed. Two methods are available.
       1) 'log_odds': This mapping first takes the log_odds of the decision
          boundaries and then passes them through a sigmoid function that
          is affected by the 'high_confidence_threshold' and the
          'confidence_mapping_slope' parameters.
       2) 'belief': This mapping only applies a linear transformation to
          the belief bounds. The g bounds are first transformed as follows:
          the belief bound for the correct choice, gc, is transformed as
          gc = 2*gc-1. The belief bound for the incorrect choice, gi, is
          transformed as gi = 1-2*gi. These transformed bounds are then
          passed through the linear transformation:
          confidence_map_slope*(g - high_confidence_threshold). The values
          are then clipped to the interval [0,1], i.e. values greater than
          1 are converted to 1, and the values smaller than 0 are converted
          to 0.
       The default method is 'log_odds'. Be aware that, the mapping method
       is only added to the saved filename for the methods different than
       'log_odds'.
 '-bs' or '--binary_split_method': A string to identify the method used to
                          binarize the subjectSession's confidence
                          reports. Available methods are 'median',
                          'half' and 'mean'. If 'median', every report
                          below the median confidence report is
                          interpreted as low confidence, and the rest
                          are high confidence. If 'half', every report
                          below 0.5 is cast as low confidence. If
                          'mean', every report below the mean confidence
                          is cast as low confidence. [Default 'median']
 '--fits_path': The path to the directory where the fit results should
                be saved or loaded from. This path can be absolute or
                relative. Be aware that the default is relative to the
                current directory. [Default 'fits_cognition']
 
 The following argument values must be supplied as JSON encoded strings.
 JSON dictionaries are written as '{"key":val,"key2":val2}'
 JSON arrays (converted to python lists) are written as '[val1,val2,val3]'
 Note that the single quotation marks surrounding the brackets, and the
 double quotation marks surrounding the keys are mandatory. Furthermore,
 if a key value should be a string, it must also be enclosed in double
 quotes.
 
 '--fixed_parameters': A dictionary of fixed parameters. The dictionary must be written as
                       '{"fixed_parameter_name":fixed_parameter_value,...}'. For example,
                       '{"cost":0.2,"dead_time":0.5}'. Note that the value null
                       can be passed as a fixed parameter. In that case, the
                       parameter will be fixed to its default value or, if the flag
                       -f is also supplied, to the parameter value loaded from the
                       previous fitting round.
                       Default depends on the method. If the method is full, the
                       confidence parameters will be fixed, and in fact ignored.
                       If the method is confidence_only, the decision parameters
                       are fixed to their default values.
 
 '--start_point': A dictionary of starting points for the fitting procedure.
                  The dictionary must be written as '{"parameter_name":start_point_value,etc}'.
                  If a parameter is omitted, its default starting value is used. You only need to specify
                  the starting points for the parameters that you wish not to start at the default
                  start point. Default start points are estimated from the subjectSession data.
 
 '-bo' or '--bounds': A dictionary of lower and upper bounds in parameter space.
             The dictionary must be written as '{"parameter_name":[low_bound_value,up_bound_value],etc}'
             As for the --start_point option, if a parameter is omitted, its default bound is used.
             Default bounds are:
             '{"cost":[0.,10],"dead_time":[0.,0.4],"dead_time_sigma":[0.,3.],
               "phase_out_prob":[0.,1.],"internal_var":[self._internal_var*1e-6,self._internal_var*1e3],
               "high_confidence_threshold":[0.,3.],"confidence_map_slope":[0.,1e12]}'
 
 '--dmKwargs': A dictionary of optional keyword args used to construct a DecisionModel instance.
               Refer to DecisionModel in decision_model.py for posible key-value pairs. [Default '{}']
 
 '--optimizer_kwargs': A dictionary of options passed to the optimizer with a few additions.
                       If the optimizer is cma, refer to fmin in cma.py for the list of
                       posible cma options. The additional option in this case is only
                       'restarts':INTEGER that sets the number of restarts used in the cmaes fmin
                       function.
                       If 'basinhopping' is selected, refer to scipy.optimize.basinhopping for
                       a detailed list of all the available options.
                       If another scipy optimizer is selected, refer to scipy minimize for
                       posible fmin options. The additional option in this case is
                       the 'repetitions':INTEGER that sets the number of independent
                       repetitions used by repeat_minize to find the minimum.
                       [Default depends on the optimizer. If 'cma', '{"restarts":1}'.
                       If 'basinhopping', '{"stepsize":0.25,"minimizer_kwargs":{"method":"Nelder-Mead"},"T":10.,"niter":100,"interval":10}.
                       If not 'cma' or 'basinhopping', '{"disp": False, "maxiter": 1000, "maxfev": 10000, "repetitions": 10}']
                       Note that the basinhopping can also accept options 'take_step', 'accept_test'
                       and 'callback' which must be callable. To achieve this functionality
                       you must pass the callable's full string definition with proper
                       indentation. This string will be evaled and the returned value
                       will be used to set the callable. Keep in mind that at the moment
                       the string is evaled, 4 variables will be available:
                       self: the Fitter instance
                       start_point: a numpy array with the starting points for the fitted parameters,
                       bounds: a 2D numpy array of shape (2,len(start_point)). bounds[0]
                               is the lower bound and bounds[1] is the upper bound.
                       optimizer_kwargs: a dict with the optimizer keyword arguments
                       A default take_step and accept_test method is used. The latter
                       only checks if the basinhopping's solution is within the bounds.
                       The former makes normally distributed steps with standard
                       deviation equal to stepsize*(bounds[1]-bounds[0]).
 
 '-f' or '--start_point_from_fit_output': A flag that tells the script to set the unspecified start_points
                       equal to the results of a previously saved fitting round. After the flag
                       the user must pass a dictionary of the form:
                       '{"method":"value","optimizer":"value","suffix":"value","cmapmeth":"value"}'
                       where the values must be the corresponding method, optimizer,
                       suffix and confidence_mapping_method used by the
                       previous fitting round. The script will then try to load the fitted parameters
                       from the file:
                       {fits_path}/{experiment}_fit_{method}_subject_{name}_session_{session}_{optimizer}{suffix}.pkl
                       or {fits_path}/{experiment}_fit_{method}_subject_{name}_session_{session}_{optimizer}_cmapmeth_{cmapmeth}{suffix}.pkl
                       depending on the cmapmeth value. The fits_path value will
                       be the one supplied with the option --fits_path.
                       The experiment, name and session are taken from the subjectSession that
                       is currently being fitted, and the method, optimizer and suffix are the
                       values passed in the previously mentioned dictionary.
  
 Example:
 python moving_bounds_fits.py -t 1 -n 1 --save"""
	options =  {'task':1,'ntasks':1,'task_base':1,'method':'full','optimizer':'cma','save':False,
				'show':False,'fit':True,'suffix':'','rt_cutoff':None,
				'merge':None,'fixed_parameters':{},'dmKwargs':{},'start_point':{},'bounds':{},
				'optimizer_kwargs':{},'experiment':'all','debug':False,'confidence_partition':100,
				'plot_merge':None,'verbose':False,'save_plot_handler':False,'load_plot_handler':False,
				'start_point_from_fit_output':None,'override':False,'plot_handler_rt_cutoff':None,
				'confidence_mapping_method':'log_odds','plot_binary':None,'save_stats':False,
				'binary_split_method':'median','fits_path':'fits_cognition'}
	if '-g' in sys.argv or '--debug' in sys.argv:
		options['debug'] = True
		logging.basicConfig(level=logging.DEBUG)
	elif '-v' in sys.argv or '--verbose' in sys.argv:
		options['verbose'] = True
		logging.disable(logging.DEBUG)
		logging.basicConfig(level=logging.INFO)
	else:
		logging.disable(logging.INFO)
		logging.basicConfig(level=logging.WARNING)
	expecting_key = True
	json_encoded_key = False
	key = None
	for i,arg in enumerate(sys.argv[1:]):
		package_logger.debug('Argument {0} found in position {1}'.format(arg,i))
		if expecting_key:
			if arg=='-t' or arg=='--task':
				key = 'task'
				expecting_key = False
			elif arg=='-nt' or arg=='--ntasks':
				key = 'ntasks'
				expecting_key = False
			elif arg=='-tb' or arg=='--task_base':
				key = 'task_base'
				expecting_key = False
			elif arg=='-m' or arg=='--method':
				key = 'method'
				expecting_key = False
			elif arg=='-o' or arg=='--optimizer':
				key = 'optimizer'
				expecting_key = False
			elif arg=='-s' or arg=='--save':
				options['save'] = True
			elif arg=='--save_plot_handler':
				options['save_plot_handler'] = True
			elif arg=='--load_plot_handler':
				options['load_plot_handler'] = True
			elif arg=='--save_stats':
				options['save_stats'] = True
			elif arg=='--show':
				options['show'] = True
			elif arg=='-g' or arg=='--debug':
				continue
			elif arg=='-v' or arg=='--verbose':
				continue
			elif arg=='--fit':
				options['fit'] = True
			elif arg=='--no-fit':
				options['fit'] = False
			elif arg=='-w':
				options['override'] = True
			elif arg=='--confidence_partition':
				key = 'confidence_partition'
				expecting_key = False
			elif arg=='-sf' or arg=='--suffix':
				key = 'suffix'
				expecting_key = False
			elif arg=='--fits_path':
				key = 'fits_path'
				expecting_key = False
			elif arg=='--rt_cutoff':
				key = 'rt_cutoff'
				expecting_key = False
			elif arg=='--plot_handler_rt_cutoff':
				key = 'plot_handler_rt_cutoff'
				expecting_key = False
			elif arg=='-e' or arg=='--experiment':
				key = 'experiment'
				expecting_key = False
			elif arg=='--fixed_parameters':
				key = 'fixed_parameters'
				expecting_key = False
				json_encoded_key = True
			elif arg=='--start_point':
				key = 'start_point'
				expecting_key = False
				json_encoded_key = True
			elif arg=='-bo' or arg=='--bounds':
				key = 'bounds'
				expecting_key = False
				json_encoded_key = True
			elif arg=='--dmKwargs':
				key = 'dmKwargs'
				expecting_key = False
				json_encoded_key = True
			elif arg=='--optimizer_kwargs':
				key = 'optimizer_kwargs'
				expecting_key = False
				json_encoded_key = True
			elif arg=='-f' or arg=='--start_point_from_fit_output':
				key = 'start_point_from_fit_output'
				expecting_key = False
				json_encoded_key = True
			elif arg=='--plot_merge':
				key = 'plot_merge'
				expecting_key = False
			elif arg=='-hcm' or arg=='--confidence_mapping_method':
				key = 'confidence_mapping_method'
				expecting_key = False
			elif arg=='--plot_binary':
				key = 'plot_binary'
				expecting_key = False
			elif arg=='-bs' or arg=='--binary_split_method':
				key = 'binary_split_method'
				expecting_key = False
			elif arg=='-h' or arg=='--help':
				print(script_help)
				sys.exit(2)
			else:
				raise RuntimeError("Unknown option: {opt} encountered in position {pos}. Refer to the help to see the list of options".format(opt=arg,pos=i+1))
		else:
			expecting_key = True
			if key in ['task','ntasks','task_base','confidence_partition']:
				options[key] = int(arg)
			elif key in ['rt_cutoff','plot_handler_rt_cutoff']:
				options[key] = float(arg)
			elif json_encoded_key:
				options[key] = json.loads(arg)
				json_encoded_key = False
			elif key in ['plot_binary']:
				options[key] = eval(arg)
			else:
				options[key] = arg
	if options['debug']:
		options['debug'] = True
		options['optimizer_kwargs']['disp'] = True
	elif options['verbose']:
		options['optimizer_kwargs']['disp'] = True
	else:
		options['optimizer_kwargs']['disp'] = False
	if not expecting_key:
		raise RuntimeError("Expected a value after encountering key '{0}' but no value was supplied".format(arg))
	if options['task_base'] not in [0,1]:
		raise ValueError('task_base must be either 0 or 1')
	# Shift task from 1 base to 0 based if necessary
	options['task']-=options['task_base']
	if options['method'] not in ['full','confidence_only','full_confidence','binary_confidence_only','full_binary_confidence']:
		raise ValueError("Unknown supplied method: '{method}'. Available values are full, confidence_only full_confidence, binary_confidence_only and full_binary_confidence".format(method=options['method']))
	options['experiment'] = options['experiment'].lower()
	if options['experiment'] not in ['all','luminancia','2afc','auditivo']:
		raise ValueError("Unknown experiment supplied: '{0}'. Available values are 'all', 'luminancia', '2afc' and 'auditivo'".format(options['experiment']))
	else:
		# Switching case to the data_io_cognition case sensitive definition of each experiment
		options['experiment'] = {'all':'all','luminancia':'Luminancia','2afc':'2AFC','auditivo':'Auditivo'}[options['experiment']]
	
	if options['plot_merge'] not in [None,'subjects','sessions','all']:
		raise ValueError("Unknown plot_merge supplied: '{0}'. Available values are None, 'subjects', 'sessions' and 'all'.".format(options['plot_merge']))
	
	if not options['start_point_from_fit_output'] is None:
		keys = options['start_point_from_fit_output'].keys()
		if (not 'method' in keys) or (not 'optimizer' in keys) or (not 'suffix' in keys) or (not 'cmapmeth' in keys):
			raise ValueError("The supplied dictionary for 'start_point_from_fit_output' does not contain the all the required keys: 'method', 'optimizer', 'suffix' and 'cmapmeth'")
	
	if options['confidence_mapping_method'].lower() not in ['log_odds','belief']:
		raise ValueError("The supplied confidence_mapping_method is unknown. Available values are 'log_odds' and 'belief'. Got {0} instead.".format(options['confidence_mapping_method']))
	else:
		options['confidence_mapping_method'] = options['confidence_mapping_method'].lower()
	
	if not os.path.isdir(options['fits_path']):
		raise ValueError('Supplied an invalid fits_path value: {0}. The fits_path must be an existing directory.'.format(options['fits_path']))
	
	package_logger.debug('Parsed options: {0}'.format(options))
	
	return options

def prepare_fit_args(fitter,options,fname):
	temp = load_Fitter_from_file(fname)
	loaded_parameters = temp.get_parameters_dict_from_fit_output(temp._fit_output)
	for k in loaded_parameters.keys():
		if not k in temp.get_fitted_parameters():
			del loaded_parameters[k]
	package_logger.debug('Loaded parameters: {0}'.format(loaded_parameters))
	if fitter.method=='full':
		start_point = loaded_parameters.copy()
		fixed_parameters = loaded_parameters.copy()
		try:
			del fixed_parameters['cost']
		except KeyError:
			pass
		try:
			del fixed_parameters['dead_time']
		except KeyError:
			pass
		try:
			del fixed_parameters['dead_time_sigma']
		except KeyError:
			pass
		try:
			del fixed_parameters['internal_var']
		except KeyError:
			pass
		try:
			del fixed_parameters['phase_out_prob']
		except KeyError:
			pass
		try:
			del start_point['high_confidence_threshold']
		except KeyError:
			pass
		try:
			del start_point['confidence_map_slope']
		except KeyError:
			pass
	elif fitter.method=='confidence_only':
		fixed_parameters = loaded_parameters.copy()
		start_point = loaded_parameters.copy()
		try:
			del start_point['cost']
		except KeyError:
			pass
		try:
			del start_point['dead_time']
		except KeyError:
			pass
		try:
			del start_point['dead_time_sigma']
		except KeyError:
			pass
		try:
			del start_point['internal_var']
		except KeyError:
			pass
		try:
			del start_point['phase_out_prob']
		except KeyError:
			pass
		try:
			del fixed_parameters['high_confidence_threshold']
		except KeyError:
			pass
		try:
			del fixed_parameters['confidence_map_slope']
		except KeyError:
			pass
	elif fitter.method=='binary_confidence_only':
		fixed_parameters = loaded_parameters.copy()
		start_point = loaded_parameters.copy()
		try:
			del start_point['cost']
		except KeyError:
			pass
		try:
			del start_point['dead_time']
		except KeyError:
			pass
		try:
			del start_point['dead_time_sigma']
		except KeyError:
			pass
		try:
			del start_point['internal_var']
		except KeyError:
			pass
		try:
			del start_point['phase_out_prob']
		except KeyError:
			pass
		try:
			del fixed_parameters['high_confidence_threshold']
		except KeyError:
			pass
	elif fitter.method=='full_binary_confidence':
		start_point = loaded_parameters.copy()
		fixed_parameters = loaded_parameters.copy()
		try:
			del fixed_parameters['cost']
		except KeyError:
			pass
		try:
			del fixed_parameters['dead_time']
		except KeyError:
			pass
		try:
			del fixed_parameters['dead_time_sigma']
		except KeyError:
			pass
		try:
			del fixed_parameters['internal_var']
		except KeyError:
			pass
		try:
			del fixed_parameters['phase_out_prob']
		except KeyError:
			pass
		try:
			del start_point['high_confidence_threshold']
		except KeyError:
			pass
	else:
		start_point = loaded_parameters.copy()
		fixed_parameters = {}
	
	for k in options['fixed_parameters'].keys():
		if options['fixed_parameters'][k] is None:
			try:
				fixed_parameters[k] = start_point[k]
			except:
				fixed_parameters[k] = None
		else:
			fixed_parameters[k] = options['fixed_parameters'][k]
	start_point.update(options['start_point'])
	if fitter.method=='full_binary_confidence' or fitter.method=='binary_confidence_only':
		try:
			del start_point['confidence_map_slope']
		except KeyError:
			pass
		fixed_parameters['confidence_map_slope'] = np.inf
	package_logger.debug('Prepared fixed_parameters = {0}'.format(fixed_parameters))
	package_logger.debug('Prepared start_point = {0}'.format(start_point))
	return fixed_parameters,start_point

if __name__=="__main__":
	# Parse input from sys.argv
	options = parse_input()
	task = options['task']
	ntasks = options['ntasks']
	
	# Prepare subjectSessions list
	subjects = io.filter_subjects_list(io.unique_subject_sessions(),'all_sessions_by_experiment')
	if options['experiment']!='all':
		subjects = io.filter_subjects_list(subjects,'experiment_'+options['experiment'])
	package_logger.debug('Total number of subjectSessions listed = {0}'.format(len(subjects)))
	package_logger.debug('Total number of subjectSessions that will be fitted = {0}'.format(len(range(task,len(subjects),ntasks))))
	fitter_plot_handler = None
	
	# Main loop over subjectSessions
	for i,s in enumerate(subjects):
		package_logger.debug('Enumerated {0} subject {1}'.format(i,s.get_key()))
		if (i-task)%ntasks==0:
			package_logger.info('Task will execute for enumerated {0} subject {1}'.format(i,s.get_key()))
			# Fit parameters if the user did not disable the fit flag
			if options['fit']:
				package_logger.debug('Flag "fit" was True')
				fitter = Fitter(s,method=options['method'],
					   optimizer=options['optimizer'],decisionModelKwArgs=options['dmKwargs'],
					   suffix=options['suffix'],rt_cutoff=options['rt_cutoff'],
					   confidence_partition=options['confidence_partition'],
					   confidence_mapping_method=options['confidence_mapping_method'],
					   binary_split_method=options['binary_split_method'],
					   fits_path=options['fits_path'])
				fname = fitter.get_save_file_name()
				if options['override'] or not (os.path.exists(fname) and os.path.isfile(fname)):
					# Set start point and fixed parameters to the user supplied values
					# Or to the parameters loaded from a previous fit round
					if options['start_point_from_fit_output']:
						package_logger.debug('Flag start_point_from_fit_output was present. Will load parameters from previous fit round')
						loaded_method = options['start_point_from_fit_output']['method']
						loaded_optimizer = options['start_point_from_fit_output']['optimizer']
						loaded_suffix = options['start_point_from_fit_output']['suffix']
						loaded_cmapmeth = options['start_point_from_fit_output']['cmapmeth']
						fname = Fitter_filename(experiment=s.experiment,method=loaded_method,name=s.get_name(),\
												session=s.get_session(),optimizer=loaded_optimizer,suffix=loaded_suffix,\
												confidence_map_method=loaded_cmapmeth,fits_path=options['fits_path'])
						package_logger.debug('Will load parameters from file: {0}'.format(fname))
						fixed_parameters,start_point = prepare_fit_args(fitter,options,fname)
					else:
						fixed_parameters = options['fixed_parameters']
						start_point = options['start_point']
					bounds = options['bounds']
					
					# Perform fit and save fit output
					fit_output = fitter.fit(fixed_parameters=fixed_parameters,\
											start_point=start_point,\
											bounds=bounds,\
											optimizer_kwargs=options['optimizer_kwargs'])
					fitter.save()
				else:
					package_logger.warning('File {0} already exists, will skip enumerated subject {1} whose key is {2}. If you wish to override saved Fitter instances, supply the flag -w.'.format(fname,i,s.get_key()))
			
			# Store stats is necessary
			if options['save_stats']:
				fname = Fitter_filename(experiment=s.experiment,method=options['method'],name=s.get_name(),
							session=s.get_session(),optimizer=options['optimizer'],suffix=options['suffix'],
							confidence_map_method=options['high_confidence_mapping_method'],
							fits_path=options['fits_path'])
				package_logger.debug('save_stats flag was true. Will load Fitter instance from file {0} and compute stats.'.format(fname))
				# Try to load the fitted data from file 'fname' or continue to next subject
				try:
					fitter = load_Fitter_from_file(fname)
					loaded_fitter = True
				except:
					package_logger.warning('Failed to load fitter from file {0}. Will continue to next subject.'.format(fname))
					loaded_fitter = False
				if loaded_fitter:
					fname = fname.replace('.pkl','_stats.pkl')
					if options['override'] or not (os.path.exists(fname) and os.path.isfile(fname)):
						with open(fname,'w') as f:
							package_logger.debug('Computing stats and storing them in file "{0}".'.format(fname))
							pickle.dump(f,fitter.stats(return_mean_rt=True,
													   return_mean_confidence=True,
													   return_median_rt=True,
													   return_median_confidence=True,
													   return_std_rt=True,
													   return_std_confidence=True,
													   return_auc=True))
					else:
						package_logger.warning('File {0} already exists, will skip stats computation for enumerated subject {1} whose key is {2}. If you wish to override saved a Fitter instance stats, supply the flag -w.'.format(fname,i,s.get_key()))
			
			# Prepare plotable data
			if options['show'] or options['save'] or options['save_plot_handler']:
				package_logger.debug('show, save or save_plot_fitter flags were True.')
				if options['load_plot_handler']:
					fname = Fitter_filename(experiment=s.experiment,method=options['method'],name=s.get_name(),
							session=s.get_session(),optimizer=options['optimizer'],suffix=options['suffix'],
							confidence_map_method=options['confidence_mapping_method'],
							fits_path=options['fits_path']).replace('.pkl','_plot_handler.pkl')
					package_logger.debug('Loading Fitter_plot_handler from file={0}'.format(fname))
					try:
						f = open(fname,'r')
						temp = pickle.load(f)
						f.close()
					except:
						package_logger.warning('Failed to load Fitter_plot_handler from file={0}. Will continue to next subject.'.format(fname))
						continue
				else:
					fname = Fitter_filename(experiment=s.experiment,method=options['method'],name=s.get_name(),
							session=s.get_session(),optimizer=options['optimizer'],suffix=options['suffix'],
							confidence_map_method=options['confidence_mapping_method'],
							fits_path=options['fits_path'])
					# Try to load the fitted data from file 'fname' or continue to next subject
					fitter = load_Fitter_from_file(fname)
					try:
						package_logger.debug('Attempting to load fitter from file "{0}".'.format(fname))
						fitter = load_Fitter_from_file(fname)
					except:
						package_logger.warning('Failed to load fitter from file {0}. Will continue to next subject.'.format(fname))
						continue
					# Create Fitter_plot_handler for the loaded Fitter instance
					package_logger.debug('Getting Fitter_plot_handler with merge_plot={0}.'.format(options['plot_merge']))
					if not options['plot_handler_rt_cutoff'] is None:
						if s.experiment=='Luminancia':
							cutoff = np.min([1.,options['plot_handler_rt_cutoff']])
						else:
							cutoff = options['plot_handler_rt_cutoff']
						package_logger.debug('Fitter_plot_handler will use rt_cutoff = {0}'.format(cutoff))
						edges = np.linspace(0,cutoff,51)
					else:
						edges = None
					temp = fitter.get_fitter_plot_handler(merge=options['plot_merge'],edges=edges)
					if options['save_plot_handler']:
						fname = fname.replace('.pkl','_plot_handler.pkl')
						if options['override'] or not (os.path.exists(fname) and os.path.isfile(fname)):
							package_logger.debug('Saving Fitter_plot_handler to file={0}.'.format(fname))
							temp.save(fname)
						else:
							package_logger.warning('Could not save Fitter_plot_handler. File {0} already exists. To override supply the flag -w.'.format(fname))
				# Add the new Fitter_plot_handler to the bucket of plot handlers
				package_logger.debug('Adding Fitter_plot_handlers')
				if fitter_plot_handler is None:
					fitter_plot_handler = temp
				else:
					fitter_plot_handler+= temp
	
	# Prepare figure saver
	if options['save']:
		if task==0 and ntasks==1:
			fname = "fits_{experiment}{method}_{cmapmeth}{suffix}".format(\
					experiment=options['experiment']+'_' if options['experiment']!='all' else '',\
					method=options['method'],suffix=options['suffix'],
					cmapmeth=options['confidence_mapping_method'])
		else:
			fname = "fits_{experiment}{method}_{cmapmeth}_{task}_{ntasks}{suffix}".format(\
					experiment=options['experiment']+'_' if options['experiment']!='all' else '',\
					method=options['method'],task=task,ntasks=ntasks,suffix=options['suffix'],\
					cmapmeth=options['confidence_mapping_method'])
		if os.path.isdir("../../figs"):
			fname = "../../figs/"+fname
		if loc==Location.cluster:
			fname+='.png'
			saver = fname
		else:
			fname+='.pdf'
			saver = PdfPages(fname)
	else:
		saver = None
	# Plot and show, or plot and save depending on the flags supplied by the user
	if options['show'] or options['save']:
		package_logger.debug('Plotting results from fitter_plot_handler')
		assert not fitter_plot_handler is None, 'Could not create the Fitter_plot_handler to plot the fitter results'
		if options['plot_merge'] and options['load_plot_handler']:
			fitter_plot_handler = fitter_plot_handler.merge(options['plot_merge'])
		fitter_plot_handler.plot(saver=saver,show=options['show'],is_binary_confidence=options['plot_binary'],
								binary_split_method=options['binary_split_method'])
		if options['save']:
			package_logger.debug('Closing saver')
			saver.close()
