#!/usr/bin/python
#-*- coding: UTF-8 -*-

"""
Decision Model package

Defines the DecisionModel class, which computes the decision bounds,
belief values, maps belief and accumulated evidence, computes first
passage time probability distributions, convolutions with non-decision
time and belief to confidence mappings. This class implements the entire
decision model.

Author: Luciano Paz
Year: 2016
"""


from __future__ import division, print_function, absolute_import, unicode_literals
import numpy as np
from scipy.signal import fftconvolve
from scipy import io
from scipy import optimize
import math, copy, sys, warnings
from utils import normcdf,normcdfinv,normpdf,average_downsample
try:
	import dp
except Exception:
	print("cost_time.py requires the c++ compiled extension lib dp.so available")
	raise

class DecisionModel():
	"""
	Class that implements the dynamic programming method that optimizes
	reward rate and computes the optimal decision bounds
	"""
	def __init__(self,model_var=None,internal_var=0.,external_var=None,
				 prior_mu_mean=0.,prior_mu_var=1.,n=500,dt=1e-2,T=10.,
				 reward=1.,penalty=0.,iti=1.,tp=0.,cost=0.05,discrete_prior=None,
				 prior_var_prob=None):
		"""
		Constructor input:
		model_var = True variance rate of the process that generates samples.
			If the variance is unknown by the DecisionModel, the model_var
			can be a numpy.ndarray specifying posible values. Notice that
			in this case, the prior_var_prob must be a numpy.ndarray of the
			same shape as model_var. If model_var is None, its value is
			computed from the internal_var and external_var inputs.
		internal_var = A float that represents the internal variance rate.
			If the input model_var is None, then its value is computed based
			on internal_var and external_var.
		external_var = A float or numpy.ndarray encoding the true external
			variance underlying the sample generation. If external_var is
			a numpy.ndarray, it is assumed to represent the posible
			variances that can underly the sample generation and the
			true variance is a priori unknown. If model_var is None,
			the value of internal_var and external_var are used to set
			model_var = internal_var + external_var
		prior_mu_mean = Mean of the prior distribution on mu
		prior_mu_var = Var of the prior distribution on mu
		n = Discretization of belief space. Number of elements in g
		dt = Time steps
		T = Max time where the value is supposed to have converged already
		reward = Numerical value of reward upon success
		penalty = Numerical value of penalty upon failure
		iti = Inter trial interval
		tp = Penalty time added to iti after failure
		cost = Cost of accumulating new evidence. Can be a float or a
			numpy ndarray. See set_cost for details of the accepted costs.
		discrete_prior = can be None or a tuple like (mu_prior,weight_prior)
			mu_prior and weight_prior must be numpy 1-D arrays of the same shape
			that hold the discrete set of posible positive mu values (mu_prior) and their
			weight (weight_prior). The initialization then renormalizes the weights to
			0.5, because the prior is assumed symmetric around zero.
		"""
		if internal_var is None and external_var is None:
			self.model_var = model_var
			self.internal_var = None
			self.external_var = None
		elif internal_var is None and not model_var is None:
			self.model_var = model_var
			self.internal_var = None
			self.external_var = external_var
		elif not internal_var is None:
			self.model_var = internal_var+external_var
			self.internal_var = internal_var
			self.external_var = external_var
		if self.model_var is None:
			raise ValueError('The resulting model_var cannot be None. There are two alternatives to specify the model_var. 1) specify its value as the input "model_var". 2) specify "internal_var" and "external_var", in this case model_var=internal_var+external_var.')
		elif not(isinstance(self.model_var,float) or isinstance(self.model_var,np.ndarray)):
			raise ValueError('The resulting model_var must be a float or numpy.ndarray. Instead type(model_var)={0}'.format(type(self.model_var)))
		elif isinstance(self.model_var,np.ndarray):
			if self.model_var.ndim>1:
				raise ValueError('The resulting model_var must be a 1 dimensional numpy.ndarray')
		
		if discrete_prior:
			temp = copy.deepcopy(discrete_prior)
			self.mu_prior,self.weight_prior = temp
			self.prior_mu_mean = 0.
			self.weight_prior/=(2*np.sum(self.weight_prior))
			self.prior_mu_var = np.sum(2*self.weight_prior*self.mu_prior**2)
			self.prior_type = 2
			warnings.warn("Discrete mu prior is still an experimental feauture with buggy behavior",FutureWarning)
		else:
			self.prior_mu_mean = prior_mu_mean
			self.prior_mu_var = prior_mu_var
			self.prior_type = 1
		self.set_n(n)
		self.dt = float(dt)
		self.T = float(T)
		self.nT = int(T/dt)+1
		self.t = np.arange(0.,self.nT,dtype=np.float64)*self.dt
		self.set_cost(cost)
		self.reward = reward
		self.penalty = penalty
		self.iti = iti
		self.tp = tp
		self.rho = 0.
		if not self.known_variance():
			if prior_var_prob is None or self.model_var.shape!=prior_var_prob.shape:
				raise ValueError('When model_var is an array (unknown variance), prior_var_prob must be an array with the same shape')
			self.prior_var_prob = prior_var_prob/np.sum(prior_var_prob)
			inds = self.model_var.argsort()
			self.model_var = self.model_var[inds]
			if isinstance(self.external_var,np.ndarray) and self.external_var.shape==self.model_var.shape:
				self.external_var = self.external_var[inds]
			if isinstance(self.internal_var,np.ndarray) and self.internal_var.shape==self.model_var.shape:
				self.internal_var = self.internal_var[inds]
			self.prior_var_prob = prior_var_prob[inds]/np.sum(prior_var_prob)
		else:
			self.prior_var_prob = None
	
	def known_variance(self):
		return isinstance(self.model_var,float) or self.model_var.size==1
	
	def conjugate_mu_prior(self):
		return self.prior_type==1
	
	def __str__(self):
		if hasattr(self,'_cost_details'):
			_cost_details = self._cost_details
			if _cost_details['type']<2:
				cost = self._cost_details['details']
			else:
				cost = self.cost
		else:
			_cost_details = {'type':None,'details':None}
			cost = self.cost
		if hasattr(self,'bounds'):
			bounds = self.bounds
		else:
			bounds = None
		string = """
<{class_module}.{class_name} object at {address}>
model_var = {model_var}, internal_var = {internal_var}, external_var = {external_var},
prior_mu_mean = {prior_mu_mean}, prior_mu_var = {prior_mu_var}, prior_type = {prior_type},
dg = {dg}, n = {n}, dt = {dt}, nT = {nT}, T = {T},
reward = {reward}, penalty = {penalty}, iti = {iti}, tp = {tp}, rho = {rho},
cost_type = {cost_type}, cost = {cost},
bounds = {bounds}
		""".format(class_module=self.__class__.__module__,
					class_name=self.__class__.__name__,
					address=hex(id(self)),
					model_var=self.model_var,
					internal_var=self.internal_var,
					external_var=self.external_var,
					prior_mu_mean=self.prior_mu_mean,
					prior_mu_var=self.prior_mu_var,
					prior_type=self.prior_type,
					dg=self.dg,
					n=self.n,
					dt=self.dt,
					nT=self.nT,
					T=self.T,
					reward=self.reward,
					penalty=self.penalty,
					iti=self.iti,
					tp=self.tp,
					rho=self.rho,
					cost_type=_cost_details['type'],
					cost=cost,
					bounds=bounds)
		return string
	
	def set_n(self,n):
		self.n = int(n)
		if self.n%2==0:
			self.n+=1
		self.dg = 1./float(self.n)
		self.g = np.linspace(self.dg/2.,1.-self.dg/2.,self.n)
	def set_dt(self,dt):
		oldt = self.t
		self.dt = float(dt)
		self.nT = int(self.T/self.dt)+1
		self.t = np.arange(0.,self.nT,dtype=np.float64)*self.dt
		if self._cost_details['type']==0:
			self.set_constant_cost(self._cost_details['details'])
		elif self._cost_details['type']==1:
			self.set_polynomial_cost(self._cost_details['details'])
		else:
			self.cost = np.interp(self.t[:-1], oldt[:-1], self.cost)
	def set_T(self,T):
		self.T = float(T)
		old_nT = self.nT
		self.nT = int(self.T/self.dt)+1
		self.t = np.arange(0.,self.nT,dtype=np.float64)*self.dt
		if self._cost_details['type']==0:
			self.set_constant_cost(self._cost_details['details'])
		elif self._cost_details['type']==1:
			self.set_polynomial_cost(self._cost_details['details'])
		else:
			old_cost = self.cost
			self.cost = np.zeros_like(self.t)
			self.cost[:old_nT-1] = old_cost
			self.cost[old_nT-1:] = old_cost[-1]
	
	def copy(self):
		model_var = copy.deepcopy(self.model_var)
		out = DecisionModel(model_var=copy.deepcopy(self.model_var),
							internal_var=copy.deepcopy(self.internal_var),
							external_var=copy.deepcopy(self.external_var),
							prior_mu_mean=self.prior_mu_mean,prior_mu_var=self.prior_mu_var,
							n=self.n,dt=self.dt,T=self.T,reward=self.reward,
							penalty=self.penalty,iti=self.iti,tp=self.tp,
							cost=0,discrete_prior=copy.deepcopy(self.discrete_prior),
							prior_var_prob=self.prior_var_prob)
		out.cost = copy.deepcopy(self.cost)
		out._cost_details = copy.deepcopy(self._cost_details)
		try:
			out.value = copy.deepcopy(self.value)
		except:
			pass
		try:
			out.bounds = copy.deepcopy(self.bounds)
		except:
			pass
		return out
	
	def set_internal_var(self,internal_var):
		self.internal_var = internal_var
		self.model_var = self.external_var + self.internal_var
	def set_external_var(self,external_var):
		self.external_var = external_var
		self.model_var = self.external_var + self.internal_var
	
	def set_cost(self,cost):
		"""
		This function constructs a DecisionModel's cost array of shape
		(nT-1,).
		
		Syntax:
		self.set_cost(cost)
		
		Input:
			cost: a float or a numpy ndarray.
			 If cost is a float, self.cost set as a numpy array with all
			of its values equal to the supplied float.
			 If cost is a numpy ndarray, the cost array is constructed
			in one of two ways. If the supplied cost's shape is equal to
			(nT-1,) then the array is copied as is to the DecisionModel's
			cost array. If the shape is not equal, then the supplied array
			is assumed to hold the coefficients of a polynomial and
			the cost array is constructed as a polyval(cost,self.t[:-1]).
		
		Related functions:
		set_constant_cost, set_polynomial_cost, set_array_cost
		"""
		if isinstance(cost,np.ndarray) and not np.isscalar(cost):
			s = cost.shape
			if len(s)>1:
				raise ValueError("Cost must be a scalar or a one dimensional numpy ndarray")
			if s[0]==self.nT-1:
				self.set_cost_array(cost)
			else:
				self.set_polynomial_cost(cost)
		else:
			self.set_constant_cost(cost)
	
	def set_constant_cost(self,cost):
		"""
		self.set_constant_cost(cost)
		
		Primitive function that sets the instances cost array as
		self.cost = float(cost)*numpy.ones(self.nT-1)
		"""
		self.cost = float(cost)*np.ones(self.nT-1)
		self._cost_details = {'type':0,'details':cost}
		#~ self.shift_cost()
	
	def set_polynomial_cost(self,coefs):
		"""
		self.set_polynomial_cost(coefs)
		
		Primitive function that sets the instances cost array as
		self.cost = numpy.polyval(coefs,self.t[:-1])
		"""
		self.cost = np.polyval(coefs,self.t[:-1])
		self._cost_details = {'type':1,'details':coefs[:]}
		#~ self.shift_cost()
	
	def set_array_cost(self,cost,shift_cost=False):
		"""
		self.set_array_cost(cost)
		
		Primitive function that sets the instances cost array as
		self.cost = cost[:]
		"""
		self.cost = cost[:]
		self._cost_details = {'type':2,'details':None}
		#~ if shift_cost:
			#~ self.shift_cost()
	
	def shift_cost(self):
		"""
		self.shift_cost()
		
		Shift cost array rigidly until after the fixed_stim_duration
		"""
		index = (self.t>=self.fixed_stim_duration).nonzero()[0][0]
		self.cost[index:] = self.cost[:(self.nT-index)]
		self.cost[:index] = 0.
	
	def post_mu_var(self,t,var_ind=None):
		"""
		Bayes update of the posterior variance at time t
		
		post_mu_var(self,t,var_ind=None)
		
		Input:
			t: A float or numpy.ndarray that is the time at which the
			posterior mu variance is desired.
		"""
		if var_ind is None:
			return 1./(t/self.model_var+1./self.prior_mu_var)
		else:
			return 1./(t/self.model_var[var_ind]+1./self.prior_mu_var)
	
	def post_mu_mean(self,t,x,var_ind=None):
		"""
		Bayes update of the posterior mean at time t with cumulated sample x
		"""
		if var_ind is None:
			return (x/self.model_var+self.prior_mu_mean/self.prior_mu_var)*self.post_mu_var(t)
		else:
			return (x/self.model_var[var_ind]+self.prior_mu_mean/self.prior_mu_var)*self.post_mu_var(t,var_ind=var_ind)
	
	def x2g(self,t,x):
		"""
		Mapping from cumulated sample x at time t to belief
		
		self.x2g(t,x)
		
		Input:
			t and x: Can be floats or numpy arrays that can be broadcasted
				together.
		
		Output: A numpy array or float depending on the inputs with the
			values of the x to g mapping at times t and points x.
		
		"""
		if self.conjugate_mu_prior():
			if self.known_variance(): # Simplest case known variance conjugate mu prior
				return normcdf(self.post_mu_mean(t,x)/np.sqrt(self.post_mu_var(t)))
			else: # Unknown variance but conjugate mu prior
				num = 0.
				den = 0.
				for var_ind,mvar in enumerate(self.model_var):
					st = np.sqrt(self.post_mu_var(t,var_ind))
					pst = self.prior_var_prob[var_ind]*st
					num+=(pst*normcdf(self.post_mu_mean(t,x,var_ind)/st))
					den+=pst
				return num/den
		else: # Discrete mu prior but known variance
			num = 0.
			den = 0.
			t_over_model_var = t/self.model_var
			x_over_model_var = x/self.model_var
			for mu, weight in zip(self.mu_prior,self.weight_prior):
				alpha_expmu2t = weight*np.exp(-0.5*mu**2*t_over_model_var)
				num+= alpha_expmu2t*np.exp(-mu*x_over_model_var)
				den+= alpha_expmu2t*np.exp(mu*x_over_model_var)
			return 1./(1.+num/den)
	
	def dx2g(self,t,x):
		"""
		Derivate of the mapping from cumulated sample x at time t to belief
		
		self.dx2g(t,x)
		
		Input:
			t and x: Can be floats or numpy arrays that can be broadcasted
				together.
		
		Output: A numpy array or float depending on the inputs with the
			derivative of the x to g mapping at times t and points x.
		
		"""
		if self.conjugate_mu_prior():
			if self.known_variance(): # Simplest case known variance conjugate mu prior
				return np.exp(-0.5*self.post_mu_mean(t,x)**2/self.post_mu_var(t))*np.sqrt(self.post_mu_var(t)/(2*np.pi))/self.model_var
			else: # Unknown variance but conjugate mu prior
				num = 0.
				den = 0.
				for var_ind,mvar in enumerate(self.model_var):
					vt = self.post_mu_var(t,var_ind)
					st = np.sqrt(vt)
					p = self.prior_var_prob[var_ind]
					num+=(p*vt/mvar*np.exp(-0.5*self.post_mu_mean(t,x,var_ind)**2/vt))
					den+=p*st
				return 0.3989422804014327*num/den
		else: # Discrete mu prior but known variance
			inv_model_var = 1./self.model_var;
			t_over_model_var = t*inv_model_var;
			x_over_model_var = x*inv_model_var;
			plus = 0.
			minus = 0.
			dplus = 0.
			dminus = 0.
			for mu, weight in zip(self.mu_prior,self.weight_prior):
				alpha_expmu2t = weight*np.exp(-0.5*mu**2*t_over_model_var)
				mu_over_model_var = mu*inv_model_var
				plus+= alpha_expmu2t*np.exp(mu*x_over_model_var)
				minus+= alpha_expmu2t*np.exp(-mu*x_over_model_var)
				dplus+= mu_over_model_var*alpha_expmu2t*np.exp(mu*x_over_model_var)
				dminus+= mu_over_model_var*alpha_expmu2t*np.exp(-mu*x_over_model_var)
		return (dminus*plus+dplus*minus)/((plus+minus)**2)
	
	def g2x(self,t,g):
		"""
		Mapping from belief at time t to cumulated sample x (inverse of x2g)
		"""
		if self.conjugate_mu_prior() and self.known_variance():
			return self.model_var*(normcdfinv(g)/np.sqrt(self.post_mu_var(t))-self.prior_mu_mean/self.prior_mu_var)
		else:
			it = np.nditer([np.array(t),np.array(g),None])
			for t_i,g_i,out in it:
				f = lambda x: self.x2g(t_i,x)-g_i
				fprime = lambda x: self.dx2g(t_i,x)
				out[...] = optimize.newton(f, 0., fprime=fprime)
			return it.operands[2]
	
	def xbounds(self, tolerance=1e-12, set_rho=True, set_bounds=True, return_values=False, root_bounds=None):
		return dp.xbounds(self,tolerance=tolerance, set_rho=set_rho, set_bounds=set_bounds, return_values=return_values, root_bounds=root_bounds)
	xbounds.__doc__ = dp.xbounds.__doc__
	
	def xbounds_fixed_rho(self, rho=None, set_bounds=False, return_values=False):
		return dp.xbounds_fixed_rho(self,rho=rho, set_bounds=set_bounds, return_values=return_values)
	xbounds_fixed_rho.__doc__ = dp.xbounds_fixed_rho.__doc__
	
	def values(self, rho=None):
		return dp.values(self,rho=rho)
	values.__doc__ = dp.values.__doc__
	
	def rt(self,mu,model_var=None,bounds=None):
		return dp.rt(self,mu,model_var=model_var,bounds=bounds)
	rt.__doc__ = dp.rt.__doc__
	
	def fpt_conf_matrix(self,first_passage_time, confidence_response, confidence_partition=100):
		return dp.fpt_conf_matrix(self,first_passage_time, confidence_response, confidence_partition=confidence_partition)
	fpt_conf_matrix.__doc__ = dp.fpt_conf_matrix.__doc__
	
	def rt_confidence_pdf(self,first_passage_time, confidence_response, dead_time_convolver, confidence_partition=100):
		"""
		rt_confidence_pdf(self,first_passage_time, confidence_response, dead_time_convolver, confidence_partition=100)
		
		This method computes the joint probability of a given response time,
		confidence and performance for the supplied model parameters and
		first passage probability density.
		
		Input:
			first_passage_time: First passage time probability density.
				A numpy array of shape (2,self.nT) that can be the
				result of np.array(cost_time.DecisionModel.rt())
			confidence_response: The output from self.confidence_mapping(...)
			dead_time_convolver: The output from self.get_dead_time_convolver(...)
			confidence_partition: An int that is passed to fpt_conf_matrix
				in order to discretize the confidence response range.
		
		Output: The joint probability density. A numpy array with of shape
			(2,confidence_partition,self.nT+len(dead_time_convolver)-1).
			The first axis represents hits [0] or misses [1].
			The second axis represents time as indeces to 
			numpy.arange(out.shape[1])*self.dt
			The third axis represents the confidence as indeces to
			numpy.linspace(0,1,confidence_partition)
		
		"""
		if isinstance(dead_time_convolver,tuple):
			dead_time_convolver = dead_time_convolver[0]
		with warnings.catch_warnings():
			warnings.simplefilter("ignore",np.VisibleDeprecationWarning)
			out = fftconvolve(self.fpt_conf_matrix(first_passage_time=first_passage_time,
													confidence_response=confidence_response,
													confidence_partition=confidence_partition),
								dead_time_convolver[None,None,:],mode='full')
		out[out<0] = 0.
		out/=(np.sum(out)*self.dt)
		return out
	
	def decision_pdf(self,first_passage_pdfs,dead_time_convolver):
		"""
		self.decision_pdf(first_passage_pdfs,dead_time_convolver)
		
		This method computes the joint probability of a given response time
		and performance for the supplied model parameters and the
		first passage probability density.
		
		Input:
			first_passage_pdfs: First passage probability density. A numpy
				array of shape (2,self.nT) that can be the output of
				np.array(self.rt(...))
			dead_time_convolver: The output from
				self.get_dead_time_convolver(...)
		
		Output:
			pdf: The joint probability density. A numpy array with of shape
				(2,self.nT+len(dead_time_convolver)-1).
				The first axis represents hits [0] or misses [1].
				The second array represents time as indeces to
				numpy.arange(0,pdf.shape[1],dtype=np.float)*self.dt
		
		"""
		if isinstance(dead_time_convolver,tuple):
			dead_time_convolver = dead_time_convolver[0]
		with warnings.catch_warnings():
			warnings.simplefilter("ignore",np.VisibleDeprecationWarning)
			decision_pdfs = fftconvolve(first_passage_pdfs,dead_time_convolver[None,:],mode='full')
		decision_pdfs[decision_pdfs<0] = 0.
		decision_pdfs/=(np.sum(decision_pdfs)*self.dt)
		return decision_pdfs
	
	def binary_confidence_pdf(self,first_passage_pdfs,confidence_response,dead_time_convolver):
		"""
		self.binary_confidence_pdf(first_passage_pdfs,confidence_response,dead_time_convolver)
		
		This method computes the joint probability of a given response time,
		binary confidence and performance for the supplied model parameters and
		first passage probability density.
		
		Input:
			first_passage_pdfs: First passage probability density. A numpy
				array of shape (2,self.nT) that can be the output of
				np.array(self.rt(...))
			confidence_response: The output from self.confidence_mapping(...)
			dead_time_convolver: The output from self.get_dead_time_convolver(...)
		
		Output:
			pdf: The joint probability density. A numpy array with of shape
				(2,2,self.nT+len(dead_time_convolver)-1).
				The first axis represents hits [0] or misses [1].
				The second axis represents low [0] and high [1] confidence
				The third array represents time as indeces to
				numpy.arange(0,pdf.shape[2],dtype=np.float)*self.dt
		
		"""
		if isinstance(dead_time_convolver,tuple):
			dead_time_convolver = dead_time_convolver[0]
		phigh = confidence_response
		plow = 1.-phigh
		confidence_rt = np.concatenate((np.array(first_passage_pdfs)[:,None,:]*plow[:,None,:],np.array(first_passage_pdfs)[:,None,:]*phigh[:,None,:]),axis=1)
		with warnings.catch_warnings():
			warnings.simplefilter("ignore",np.VisibleDeprecationWarning)
			confidence_pdfs = fftconvolve(confidence_rt,dead_time_convolver[None,None,:],mode='full')
		confidence_pdfs[confidence_pdfs<0] = 0.
		confidence_pdfs/=(np.sum(confidence_pdfs)*self.dt)
		return confidence_pdfs
	
	def get_dead_time_convolver(self,dead_time,dead_time_sigma,return_conv_x=False):
		"""
		self.get_dead_time_convolver(dead_time,dead_time_sigma,return_conv_x=False):
		
		This function returns the dead time (aka non-decision time)
		distribution, which is convolved with the first passage time
		probability density to get the real response time distribution.
		
		Input:
			dead_time: A float that represents the center of the
				gaussian used as the dead time distribution (actually
				only the upper half is used)
			dead_time_sigma: A float that represents the gaussian's
				standard deviation.
			return_conv_x: A bool. If True, the convolution's
				corresponding time numpy.array is also returned.
		
		Output:
			conv_val or (conv_val,conv_x) depending on whether
			return_conv_x is True or not.
			conv_val is an array with the values of the dead time
			distribution for the times that are in conv_x.
		
		"""
		must_downsample = True
		if self.dt>1e-3:
			_dt = 1e-3
		else:
			must_downsample = False
			_dt = self.dt
		
		conv_x_T = dead_time+6*dead_time_sigma
		dense_conv_x_nT = int(conv_x_T/_dt)+1
		conv_x_nT = int(conv_x_T/self.dt)+1
		dense_conv_x = np.arange(0,dense_conv_x_nT)*_dt
		if dead_time_sigma>0:
			dense_conv_val = normpdf(dense_conv_x,dead_time,dead_time_sigma)
			dense_conv_val[dense_conv_x<dead_time] = 0.
		else:
			dense_conv_val = np.zeros_like(dense_conv_x)
			dense_conv_val[np.floor(dead_time/_dt)] = 1.
		conv_x = np.arange(0,conv_x_nT)*self.dt
		if must_downsample:
			#~ conv_val = average_downsample(dense_conv_val,conv_x_nT)
			if dense_conv_x_nT%conv_x_nT==0:
				ratio = int(np.round(dense_conv_x_nT/conv_x_nT))
			else:
				ratio = int(np.ceil(dense_conv_x_nT/conv_x_nT))
			tail = dense_conv_x_nT%ratio
			if tail!=0:
				padded_cv = np.concatenate((dense_conv_val,np.nan*np.ones(ratio-tail,dtype=np.float)),axis=0)
			else:
				padded_cv = dense_conv_val
			padded_cv = np.reshape(padded_cv,(-1,ratio))
			conv_val = np.nanmean(padded_cv,axis=1)
		else:
			conv_val = dense_conv_val
		conv_val/=np.sum(conv_val)
		if return_conv_x:
			return conv_val,conv_x
		else:
			return conv_val
	
	def refine_value(self,tolerance=1e-12,dt=None,n=None,T=None):
		"""
		This method re-computes the value of the beliefs using the
		average reward (rho) that was already computed.
		"""
		change = False
		if dt is not None:
			if dt<self.dt:
				change = True
				oldt = self.t.copy()
				self.dt = float(dt)
				self.nT = int(self.T/self.dt)+1
				self.t = np.arange(0.,self.nT,dtype=np.float64)*self.dt
				if self._cost_details['type']==0:
					self.set_constant_cost(self._cost_details['details'])
				elif self._cost_details['type']==1:
					self.set_polynomial_cost(self._cost_details['details'])
				else:
					self.cost = np.interp(self.t[:-1], oldt[:-1], self.cost)
		if T is not None:
			if T>self.T:
				change = True
				self.T = float(T)
				old_nT = self.nT
				old_cost = self.cost
				self.nT = int(self.T/self.dt)+1
				self.t = np.arange(0.,self.nT,dtype=np.float64)*self.dt
				if self._cost_details['type']==0:
					self.set_constant_cost(self._cost_details['details'])
				elif self._cost_details['type']==1:
					self.set_polynomial_cost(self._cost_details['details'])
				else:
					self.cost = np.zeros_like(self.t)
					self.cost[:old_nT-1] = old_cost
					self.cost[old_nT-1:] = old_cost[-1]
		if n is not None:
			n = int(n)
			if n%2==0:
				n+=1
			if n>self.n:
				change = True
				self.n = n
				self.g = np.linspace(0.,1.,self.n)
		if change:
			temp = self.xbounds_fixed_rho(set_bounds=True, return_values=True)
			val0 = temp[2][0,int(0.5*self.n)]
			if abs(val0)>tolerance:
				if val0<0:
					ub = self.rho
					lb = self.rho-1e-2
				else:
					ub = self.rho+1e-2
					lb = self.rho
				xbs = self.xbounds(tolerance=tolerance, set_rho=True, set_bounds=True, root_bounds=(lb,ub))
			else:
				self.value = temp[0]
				xbs = (temp[0],temp[1])
		else:
			xbs = self.belief_bound_to_x_bound()
			xbs = (xbs[0],xbs[1])
		return xbs
	
	def log_odds(self):
		ret = np.log(self.bounds)-np.log(1-self.bounds)
		ret[1]*=-1
		return ret
	
	def confidence_mapping(self,high_confidence_threshold,confidence_map_slope,confidence_mapping_method='log_odds'):
		"""
		self.high_confidence_mapping(high_confidence_threshold,confidence_map_slope)
		
		Get the high confidence mapping as a function of time.
		Returns a numpy array of shape (2,self.nT)
		The output[0] is the mapping for hits and output[1] is the
		mapping for misses.
		
		"""
		if confidence_mapping_method=='log_odds':
			return self.confidence_mapping_log_odds(high_confidence_threshold,confidence_map_slope)
		elif confidence_mapping_method=='belief':
			return self.confidence_mapping_belief(high_confidence_threshold,confidence_map_slope)
		else:
			raise ValueError('Undefined high confidence mapping method: {0}'.format(confidence_mapping_method))
	
	def confidence_mapping_log_odds(self,high_confidence_threshold,confidence_map_slope):
		"""
		self.high_confidence_mapping_log_odds(high_confidence_threshold,confidence_map_slope)
		
		Backend of self.high_confidence_mapping that implements the log_odds
		mapping. Returns the same type as self.high_confidence_mapping.
		
		"""
		if self.dt>1e-3:
			_dt = 1e-3
		else:
			_dt = None
		
		if _dt:
			_nT = int(self.T/_dt)+1
			_t = np.arange(0.,_nT,dtype=np.float64)*_dt
			log_odds = self.log_odds()
			log_odds = np.array([np.interp(_t,self.t,log_odds[0]),np.interp(_t,self.t,log_odds[1])])
		else:
			_nT = self.nT
			log_odds = self.log_odds()
			_dt = self.dt
		# Likely to raise warnings with exp overflows or invalid values in multiply
		# if confidence_map_slope is inf or log_odds==high_confidence_threshold
		# These issues are resolved naturally in the two-line statements
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			phigh = 1./(1.+np.exp(confidence_map_slope*(high_confidence_threshold-log_odds)))
		phigh[high_confidence_threshold==log_odds] = 0.5
		
		if _dt:
			if _nT%self.nT==0:
				ratio = int(np.round(_nT/self.nT))
			else:
				ratio = int(np.ceil(_nT/self.nT))
			tail = _nT%ratio
			if tail!=0:
				padded_phigh = np.concatenate((phigh,np.nan*np.ones((2,ratio-tail),dtype=np.float)),axis=1)
			else:
				padded_phigh = phigh
			padded_phigh = np.reshape(padded_phigh,(2,-1,ratio))
			phigh = np.nanmean(padded_phigh,axis=2)
		return phigh
	
	def confidence_mapping_belief(self,high_confidence_threshold,confidence_map_slope):
		"""
		self.high_confidence_mapping_belief(high_confidence_threshold,confidence_map_slope)
		
		Backend of self.high_confidence_mapping that implements the belief
		mapping. Returns the same type as self.high_confidence_mapping.
		
		"""
		if self.dt>1e-3:
			_dt = 1e-3
		else:
			_dt = None
		
		if _dt:
			_nT = int(self.T/_dt)+1
			_t = np.arange(0.,_nT,dtype=np.float64)*_dt
			belief = self.bounds.copy()
			belief = np.array([np.interp(_t,self.t,2*(belief[0]-0.5)),np.interp(_t,self.t,2*(0.5-belief[1]))])
		else:
			_nT = self.nT
			belief = self.bounds.copy()
			belief[0] = 2*(belief[0]-0.5)
			belief[1] = 2*(0.5-belief[1])
			_dt = self.dt
		
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			phigh = confidence_map_slope*(belief-high_confidence_threshold)
			phigh[np.isnan(phigh)] = 0.5
			phigh[phigh>1] = 1
			phigh[phigh<0] = 0
		
		if _dt:
			if _nT%self.nT==0:
				ratio = int(np.round(_nT/self.nT))
			else:
				ratio = int(np.ceil(_nT/self.nT))
			tail = _nT%ratio
			if tail!=0:
				padded_phigh = np.concatenate((phigh,np.nan*np.ones((2,ratio-tail),dtype=np.float)),axis=1)
			else:
				padded_phigh = phigh
			padded_phigh = np.reshape(padded_phigh,(2,-1,ratio))
			phigh = np.nanmean(padded_phigh,axis=2)
		return phigh

def diffusion_path_samples(mu,var_rate,dt,T,xb,reps=10):
	paths = []
	sigma = np.sqrt(var_rate*dt)
	if not isinstance(mu,np.ndarray):
		mus = mu*np.ones(reps)
	else:
		mus = mu
	nT = int(T/dt)+1
	for mu in mus:
		path = {'x':[0],'t':[0]}
		decided = False
		for t_i in np.arange(1,nT):
			t = t_i*dt
			stim = sigma*np.random.randn(1)+mu*dt
			path['x'].append(path['x'][-1]+stim)
			path['t'].append(t)
			if path['x'][-1]>=xb[0][t_i+1]:
				path['dec']=1
				path['rt']=t
				decided = True
				break
			elif path['x'][-1]<=xb[1][t_i+1]:
				path['dec']=2
				path['rt']=t
				decided = True
				break
		if not decided:
			path['dec']=None
			path['rt']=None
		paths.append(path)
	return sorted(paths,key=lambda path:path['rt'], reverse=True)

def sim_rt(mu,var_rate,dt,T,xb,reps=10000):
	if not isinstance(mu,np.ndarray):
		mu = mu*np.ones(reps)
	sim = np.zeros_like(mu)
	rt = np.zeros_like(mu)
	decision = np.zeros_like(mu)
	not_decided = np.ones_like(mu,dtype=np.bool)
	sigma = np.sqrt(dt*var_rate)
	nT = int(T/dt)+1
	for i in range(1,nT):
		t = float(dt)*i
		sim+= sigma*np.random.randn(*mu.shape)
		stim = sim+t*mu
		dec1 = np.logical_and(stim>=xb[0][i+1],not_decided)
		dec2 = np.logical_and(stim<=xb[1][i+1],not_decided)
		if any(dec1):
			rt[dec1] = t
			decision[dec1] = 1
			not_decided[dec1] = 0
		if any(dec2):
			rt[dec2] = t
			decision[dec2] = 2
			not_decided[dec2] = 0
		if not any(not_decided):
			break
	out = (rt,decision)
	return out

def _test():
	out = dp.testsuite()
	dict1,dict2,dict3,t,cx,cg1,cg2,cg3,cdg1,cdg2,cdg3,cx1,cx2,cx3 = out
	dict1['T'] = dict2['T'] = dict3['T'] = 3.
	d1 = DecisionModel(**dict1)
	d2 = DecisionModel(**dict2)
	d3 = DecisionModel(**dict3)
	d1.set_internal_var(10.)
	d2.set_internal_var(10.)
	d3.set_internal_var(10.)
	print(d1)
	print(d2)
	print(d3)
	from matplotlib import pyplot as plt
	from matplotlib.colors import LogNorm
	x = np.linspace(-30,30,100)
	
	dx = x[1]-x[0]
	g1 = d1.x2g(t,x)[:,None]
	g2 = d2.x2g(t,x)[:,None]
	g3 = d3.x2g(t,x)[:,None]
	dg1 = d1.dx2g(t,x)
	dg2 = d2.dx2g(t,x)
	dg3 = d3.dx2g(t,x)
	numdg1 = np.vstack(((g1[1,:]-g1[0,:])/dx,0.5*(g1[2:,:]-g1[:-2,:])/dx,(g1[-1,:]-g1[-2,:])/dx))
	numdg2 = np.vstack(((g2[1,:]-g2[0,:])/dx,0.5*(g2[2:,:]-g2[:-2,:])/dx,(g2[-1,:]-g2[-2,:])/dx))
	numdg3 = np.vstack(((g3[1,:]-g3[0,:])/dx,0.5*(g3[2:,:]-g3[:-2,:])/dx,(g3[-1,:]-g3[-2,:])/dx))
	x1 = d1.g2x(t,g1[:,0])
	x2 = d2.g2x(t,g2[:,0])
	x3 = d3.g2x(t,g3[:,0])
	plt.subplot(131)
	plt.plot(x,g1,label='conj',color='b')
	plt.plot(x,g2,label='discrete mu',color='g')
	plt.plot(x,g3,label='discrete var',color='r')
	plt.plot(cx,cg1,label='C conj',color='b',linestyle='--')
	plt.plot(cx,cg2,label='C discrete mu',color='g',linestyle='--')
	plt.plot(cx,cg3,label='C discrete var',color='r',linestyle='--')
	plt.legend(loc='best', fancybox=True, framealpha=0.5)
	plt.title('x2g')
	plt.subplot(132)
	plt.plot(x,dg1,label='conj',color='b')
	plt.plot(x,dg2,label='discrete mu',color='g')
	plt.plot(x,dg3,label='discrete var',color='r')
	plt.plot(cx,cdg1,label='C conj',color='b',linestyle='--')
	plt.plot(cx,cdg2,label='C discrete mu',color='g',linestyle='--')
	plt.plot(cx,cdg3,label='C discrete var',color='r',linestyle='--')
	plt.title('dx2g')
	plt.subplot(133)
	plt.plot(cx,x1-cx1,label='C conj',color='b',linestyle='-')
	plt.plot(cx,x2-cx2,label='C discrete mu',color='g',linestyle='-')
	plt.plot(cx,x3-cx3,label='C discrete var',color='r',linestyle='-')
	plt.title('g2x')
	plt.suptitle(r'x $\leftrightarrow$ Mapping')
	print(t)
	
	print('conj')
	xub1,xlb1,v1,ve1,_,_ = d1.xbounds(return_values=True)
	print(d1.rho)
	print('discrete mu')
	xub2,xlb2,v2,ve2,_,_ = d2.xbounds(return_values=True)
	print(d2.rho)
	print('discrete var')
	xub3,xlb3,v3,ve3,_,_ = d3.xbounds(return_values=True)
	print(d3.rho)
	plt.figure()
	plt.subplot(211)
	plt.plot(d1.t,xub1,'b',label='Conj')
	plt.plot(d1.t,xlb1,'b')
	plt.plot(d2.t,xub2,'g',label='Discrete mu')
	plt.plot(d2.t,xlb2,'g')
	plt.plot(d3.t,xub3,'r',label='Discrete var')
	plt.plot(d3.t,xlb3,'r')
	plt.legend(loc='best', fancybox=True, framealpha=0.5)
	plt.ylabel('x(t) bounds')
	plt.subplot(212)
	plt.plot(d1.t,d1.bounds[0],'b',label='Conj')
	plt.plot(d1.t,d1.bounds[1],'b')
	plt.plot(d2.t,d2.bounds[0],'g',label='Discrete mu')
	plt.plot(d2.t,d2.bounds[1],'g')
	plt.plot(d3.t,d3.bounds[0],'r',label='Discrete var')
	plt.plot(d3.t,d3.bounds[1],'r')
	plt.suptitle('Bounds')
	plt.ylabel('g(t) bounds')
	plt.xlabel('T')
	
	print('Computing rt')
	print('Conj')
	mu = 1.
	rt1 = np.array(d1.rt(mu,bounds=(xub1,xlb1)))
	print('Discrete mu')
	rt2 = np.array(d2.rt(mu,bounds=(xub2,xlb2)))
	rt3 = np.zeros_like(rt1)
	for model_var,prior_var_prob in zip(d3.model_var,d3.prior_var_prob):
		print('Discrete var: {0}'.format(model_var))
		rt3+= np.array(d3.rt(mu,bounds=(xub1,xlb1),model_var=model_var))*prior_var_prob
	plt.figure()
	plt.plot(d1.t,rt1[0],'b',label='Conj hit')
	plt.plot(d1.t,rt1[1],'--b',label='Conj miss')
	plt.plot(d2.t,rt2[0],'g',label='Discrete mu')
	plt.plot(d2.t,rt2[1],'--g')
	plt.plot(d3.t,rt3[0],'r',label='Discrete var hit')
	plt.plot(d3.t,rt3[1],'--r',label='Discrete var miss')
	plt.ylabel('RT prob')
	plt.xlabel('T')
	plt.legend(loc='best', fancybox=True, framealpha=0.5)
	plt.suptitle('First passage time')
	
	plt.figure()
	plt.subplot(311)
	plt.imshow((v1[1:]-ve1).T,aspect='auto',cmap='jet',interpolation='none',origin='lower',extent=[d1.t[0],d1.t[-1],d1.g[0],d1.g[-1]])
	plt.colorbar()
	plt.subplot(312)
	plt.imshow((v2[1:]-ve2).T,aspect='auto',cmap='jet',interpolation='none',origin='lower',extent=[d2.t[0],d2.t[-1],d2.g[0],d2.g[-1]])
	plt.colorbar()
	plt.subplot(313)
	plt.imshow((v3[1:]-ve3).T,aspect='auto',cmap='jet',interpolation='none',origin='lower',extent=[d3.t[0],d3.t[-1],d3.g[0],d3.g[-1]])
	plt.colorbar()
	plt.suptitle(r'$\tilda{V}-V_{explore}')
	
	print('Computing confidence mappings')
	
	high_confidence_threshold = 0.3
	confidence_map_slope = 1.7
	dead_time = 0.2
	dead_time_sigma = 0.4
	
	confidence_response1 = d1.confidence_mapping(high_confidence_threshold,confidence_map_slope,confidence_mapping_method='belief')
	confidence_response2 = d2.confidence_mapping(high_confidence_threshold,confidence_map_slope,confidence_mapping_method='belief')
	confidence_response3 = d3.confidence_mapping(high_confidence_threshold,confidence_map_slope,confidence_mapping_method='belief')
	
	plt.figure()
	ax = plt.subplot(211)
	plt.plot(d1.t,(d1.bounds.T*2-1)*np.array([1,-1]),label='conj')
	plt.plot(d2.t,(d2.bounds.T*2-1)*np.array([1,-1]),label='Discrete mu')
	plt.plot(d3.t,(d3.bounds.T*2-1)*np.array([1,-1]),label='Discrete var')
	plt.ylabel('Normed Belief')
	plt.subplot(212,sharex=ax)
	plt.plot(d1.t,confidence_response1.T,label='conj')
	plt.plot(d2.t,confidence_response2.T,label='Discrete mu')
	plt.plot(d3.t,confidence_response3.T,label='Discrete var')
	plt.ylabel('Confidence response')
	plt.xlabel('T')
	
	
	print('Computing first passage time confidence response partition matrix')
	
	fpt_conf1 = d1.fpt_conf_matrix(rt1,confidence_response1,100)
	fpt_conf2 = d2.fpt_conf_matrix(rt2,confidence_response2,100)
	fpt_conf3 = d3.fpt_conf_matrix(rt3,confidence_response3,100)
	plt.figure()
	ax1 = plt.subplot(231)
	plt.imshow(fpt_conf1[0].T,aspect='auto',cmap='jet',origin='lower',interpolation='none',extent=[d1.t[0],d1.t[-1],0,1])
	plt.plot(d1.t,confidence_response1[0],'--k')
	plt.title('Conj')
	ax2 = plt.subplot(232)
	plt.imshow(fpt_conf2[0].T,aspect='auto',cmap='jet',origin='lower',interpolation='none',extent=[d2.t[0],d2.t[-1],0,1])
	plt.plot(d2.t,confidence_response2[0],'--k')
	plt.title('Discrete mu')
	ax3 = plt.subplot(233)
	plt.imshow(fpt_conf3[0].T,aspect='auto',cmap='jet',origin='lower',interpolation='none',extent=[d3.t[0],d3.t[-1],0,1])
	plt.plot(d3.t,confidence_response3[0],'--k')
	plt.title('Discrete var')
	plt.subplot(234,sharex=ax1)
	plt.plot(d1.t,np.sum(fpt_conf1,axis=1).T)
	plt.plot(d1.t,rt1.T,'--')
	plt.ylabel('Prob')
	plt.xlabel('T')
	plt.subplot(235,sharex=ax2)
	plt.plot(d2.t,np.sum(fpt_conf2,axis=1).T)
	plt.plot(d2.t,rt2.T,'--')
	plt.xlabel('T')
	plt.subplot(236,sharex=ax3)
	plt.plot(d3.t,np.sum(fpt_conf3,axis=1).T)
	plt.plot(d3.t,rt3.T,'--')
	plt.xlabel('T')
	
	print('Computing dead time convolver and adding dead time to response time distributions')
	dead_time_convolver1 = d1.get_dead_time_convolver(dead_time,dead_time_sigma)
	dead_time_convolver2 = d2.get_dead_time_convolver(dead_time,dead_time_sigma)
	dead_time_convolver3 = d3.get_dead_time_convolver(dead_time,dead_time_sigma)
	rt_conf1 = d1.rt_confidence_pdf(rt1,confidence_response1,dead_time_convolver1,100)
	rt_conf2 = d2.rt_confidence_pdf(rt2,confidence_response2,dead_time_convolver2,100)
	rt_conf3 = d3.rt_confidence_pdf(rt3,confidence_response3,dead_time_convolver3,100)
	t1 = np.arange(rt_conf1.shape[1])*d1.dt
	t2 = np.arange(rt_conf2.shape[1])*d2.dt
	t3 = np.arange(rt_conf3.shape[1])*d3.dt
	plt.figure()
	ax = plt.subplot(231)
	plt.imshow(rt_conf1[0].T,aspect='auto',cmap='jet',origin='lower',interpolation='none',extent=[t1[0],t1[-1],0,1])
	ax = plt.subplot(234)
	plt.imshow(rt_conf1[1].T,aspect='auto',cmap='jet',origin='lower',interpolation='none',extent=[t1[0],t1[-1],0,1])
	ax = plt.subplot(232)
	plt.imshow(rt_conf2[0].T,aspect='auto',cmap='jet',origin='lower',interpolation='none',extent=[t2[0],t2[-1],0,1])
	ax = plt.subplot(235)
	plt.imshow(rt_conf2[1].T,aspect='auto',cmap='jet',origin='lower',interpolation='none',extent=[t2[0],t2[-1],0,1])
	ax = plt.subplot(233)
	plt.imshow(rt_conf3[0].T,aspect='auto',cmap='jet',origin='lower',interpolation='none',extent=[t3[0],t3[-1],0,1])
	ax = plt.subplot(236)
	plt.imshow(rt_conf3[1].T,aspect='auto',cmap='jet',origin='lower',interpolation='none',extent=[t3[0],t3[-1],0,1])
	
	print('Finished test suite')
	plt.show(True)

if __name__=="__main__":
	_test()
