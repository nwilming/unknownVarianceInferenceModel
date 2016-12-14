#!/usr/bin/python
#-*- coding: UTF-8 -*-
"""
Package for loading the behavioral dataset

Defines the SubjectSession class that provides an interface to load
the experimental data from the supplied raw_data_dir.

Author: Luciano Paz
Year: 2016
"""

from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as np
from scipy import io as io
import os, itertools, sys, random, re, scipy.integrate, logging

package_logger = logging.getLogger("data_io_cognition")

class SubjectSession:
	def __init__(self,name,session,experiment,data_dir):
		"""
		SubjectSession(self,name,session,experiment,data_dir)
		
		This class is an interface to flexible load the data from
		Ais et al 2016 of all 3 2AFC tasks. Upon creation, a list of
		subject names, a list of sessions and a single experiment name
		is provided along with a list of directory paths where the
		corresponding data experimental data is located.
		This class can then be used to load the data of the different
		experiments into a standarized numpy array format.
		
		Input:
			'name': A string or a list of strings that represent subject
				names. These names can be arbitrary and unrelated to
				the real name of the subject. In fact, the package
				data_io_cognition provides a function to anonimize the
				subjects
			'session': An int or list of ints with the sessions that
				should be loaded by the created SubjectSession instance.
			'experiment': One of the two alternative forced choice
				stardarized experiment names. The contrast task is named
				'2AFC', the auditory task is named 'Auditivo' and the
				luminance task is named 'Luminancia'. All experiment
				names are case sensitive.
			'data_dir': A string or list of strings that are paths to
				the directories where to look for each of the subject's
				names data files. If 'data_dir' is a list, 'name' must
				also be a list of the same len.
		
		"""
		package_logger.debug('Creating SubjectSession instance')
		self.logger = logging.getLogger("data_io_cognition.SubjectSession")
		self.logger.debug('Inputs name:{name}, session: {session}, experiment: {experiment}, data_dir: {data_dir}'.format(
						name=name,session=session,experiment=experiment,data_dir=data_dir))
		try:
			self.session = int(session)
			self._single_session = True
		except:
			self.session = [int(s) for s in session]
			self._single_session = False
		self.logger.debug('Is single session? {0}'.format(self._single_session))
		self.logger.debug("Instance's session: {0}".format(self.session))
		self.experiment = str(experiment)
		self.logger.debug("Instance's experiment: {0}".format(self.experiment))
		if isinstance(data_dir,list):
			self.name = [str(n) for n in name]
			self.data_dir = [str(d) for d in data_dir]
			if len(self.name)!=len(self.data_dir):
				raise ValueError('The data_dir and name lists must have the same number of elements')
			self._single_data_dir = False
			self._map_data_dir_name = {}
			for n,d in zip(self.name,self.data_dir):
				self._map_data_dir_name[d] = n
		else:
			if isinstance(name,list):
				raise TypeError('The name input cannot be a list if the supplied data_dir is string')
			self.name = str(name)
			self.data_dir = str(data_dir)
			self._map_data_dir_name = {self.data_dir:self.name}
			self._single_data_dir = True
		self.logger.debug('Is single data_dir? {0}'.format(self._single_data_dir))
		self.logger.debug("Instance's name: {0}".format(self.name))
		self.logger.debug("Instance's data_dir: {0}".format(self.data_dir))
		self.logger.debug("Instance's _map_data_dir_name: {0}".format(self._map_data_dir_name))
	
	def get_name(self):
		"""
		self.get_name()
		
		Returns a string with the subjectSession name properly converted
		to a string.
		
		"""
		if self._single_data_dir:
			name = str(self.name)
		else:
			name = '['+'-'.join([str(n) for n in self.name])+']'
		return name
	
	def get_session(self):
		"""
		self.get_session()
		
		Returns a string with the subjectSession session properly
		converted to a string.
		
		"""
		if self._single_session:
			session = str(self.session)
		else:
			session = '['+'-'.join([str(s) for s in self.session])+']'
		return session
	
	def get_key(self):
		"""
		self.get_key()
		
		Returns a string with the subjectSession key:
		{experiment}_name={name}_session={session}
		where {name} is taken from self.get_name() and {session} is
		taken from self.get_session()
		
		"""
		return '{experiment}_name={name}_session={session}'.format(experiment=self.experiment,name=self.get_name(),session=self.get_session())
	
	def get_name_from_data_dir(self,data_dir,override_raw_data_dir=None):
		"""
		self.get_name_from_data_dir(data_dir,override_raw_data_dir=None)
		
		Returns the name string that corresponds to the supplied data_dir.
		
		Optional input override_raw_data_dir:
		Because sometimes the subjectSessions are constructed in other
		systems and the data_dir may be different in the running system
		it is posible to provide an override to the data_dir.
		If override_raw_data_dir is not None, then it must be a dict
		with keys 'replacement' and 'original'. The data dir is then
		replaced as follows:
		data_dir.replace(override_raw_data_dir['replacement'],override_raw_data_dir['original'])
		
		"""
		if override_raw_data_dir is None:
			return self._map_data_dir_name[data_dir]
		else:
			return self._map_data_dir_name[data_dir.replace(override_raw_data_dir['replacement'],override_raw_data_dir['original'])]
	
	def change_name(self,new_name,orig=None):
		"""
		self.change_name(new_name,orig=None)
		
		Change the subjectSession name with the supplied new_name string.
		If the subjectSession has a list of names associated to it, it
		is mandatory to supply the original name which is going to be
		replaced in the input 'orig'.
		
		"""
		if not self._single_data_dir:
			if orig is None:
				raise ValueError('Must supply the original name value when the SubjectSession has more than one data_dir')
			changed_index = self.name.index(orig)
			self.name[changed_index] = new_name
			self._map_data_dir_name[self.data_dir[changed_index]] = new_name
		else:
			self.name = new_name
			self._map_data_dir_name[self.data_dir] = new_name
	
	def list_data_files(self,override_raw_data_dir=None):
		"""
		self.list_data_files(override_raw_data_dir=None)
		
		Returns a list of the data files in the data_dir paths.
		The override_raw_data_dir can be used to replace portions of the
		data_dir path. This feature is included for situations where
		the subjectSession was created in a different path structure.
		To do this, override_raw_data_dir must be a dict with keys
		'original' and 'replacement'. Each of the data_dirs is replaced
		as follows:
		data_dir.replace(override_raw_data_dir['original'],override_raw_data_dir['replacement'])
		
		"""
		if self._single_data_dir:
			if override_raw_data_dir:
				data_dir = self.data_dir.replace(override_raw_data_dir['original'],override_raw_data_dir['replacement'])
			else:
				data_dir = self.data_dir
			return [os.path.join(data_dir,f) for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir,f))]
		else:
			listdirs = []
			for dd in self.data_dir:
				if override_raw_data_dir:
					dd = dd.replace(override_raw_data_dir['original'],override_raw_data_dir['replacement'])
				listdirs.extend([os.path.join(dd,f) for f in os.listdir(dd) if os.path.isfile(os.path.join(dd,f))])
			return list(set(listdirs))
	
	def iter_data(self,override_raw_data_dir=None):
		"""
		self.iter_data(override_raw_data_dir=None)
		
		Iterate over the experimental data files listed in the data_dirs.
		override_raw_data_dir is provided for situations where the
		subjectSession was created in a different path structure. To do
		this, override_raw_data_dir must be a dict with keys 'original'
		and 'replacement'. Each of the data_dirs is replaced as follows:
		data_dir.replace(override_raw_data_dir['original'],override_raw_data_dir['replacement'])
		
		Output:
			Each yielded value is a 2D numpy.ndarray. axis=0 represent
			different trials and axis=1 different data for each trial.
			Refer to self.column_description to get a description of
			the data contained at each index of axis=1.
		
		"""
		if self.experiment=='Luminancia':
			if self._single_session:
				data_files = [f for f in self.list_data_files(override_raw_data_dir) if ((int(re.search('(?<=_B)[0-9]+(?=_)',f).group())-1)//4+1)==self.session and f.endswith('.mat')]
			else:
				data_files = [f for f in self.list_data_files(override_raw_data_dir) if ((int(re.search('(?<=_B)[0-9]+(?=_)',f).group())-1)//4+1) in self.session and f.endswith('.mat')]
			sessions = [((int(re.search('(?<=_B)[0-9]+(?=_)',f).group())-1)//4+1) for f in data_files]
			for f,session in zip(data_files,sessions):
				aux = io.loadmat(f)
				mean_target_lum = aux['trial'][:,1]
				rt = aux['trial'][:,5]*1e-3 # Convert to seconds
				performance = aux['trial'][:,7] # 1 for success, 0 for fail
				confidence = aux['trial'][:,8] # 2 for high confidence, 1 for low confidence
				if aux['trial'].shape[1]>9:
					selected_side = aux['trial'][:,9];
				else:
					selected_side = np.nan*np.ones_like(rt)
				name = self.get_name_from_data_dir(os.path.dirname(f),override_raw_data_dir)
				if isinstance(name,int):
					data_matrix = np.array([mean_target_lum,rt,performance,confidence,selected_side,
										name*np.ones_like(rt),session*np.ones_like(rt)]).squeeze().T
				else:
					data_matrix = np.array([mean_target_lum,rt,performance,confidence,selected_side,
										session*np.ones_like(rt)]).squeeze().T
				yield data_matrix
		elif self.experiment=='2AFC':
			if self._single_session:
				data_files = [f for f in self.list_data_files(override_raw_data_dir) if int(re.search('(?<=sesion)[0-9]+',f).group())==self.session and f.endswith('.txt')]
			else:
				data_files = [f for f in self.list_data_files(override_raw_data_dir) if int(re.search('(?<=sesion)[0-9]+',f).group()) in self.session and f.endswith('.txt')]
			sessions = [int(re.search('(?<=sesion)[0-9]+',f).group()) for f in data_files]
			for f,session in zip(data_files,sessions):
				selected_side, performance, rt, contraste, confidence, phase, orientation = np.loadtxt(f, delimiter=' ', unpack=True)
				# 2AFC has too much resolution in the variable 'contraste'
				# and it slows down the fitting procedure. We will
				# coarse the data in the following statement
				contraste = np.round(contraste*5e3)/5e3
				name = self.get_name_from_data_dir(os.path.dirname(f),override_raw_data_dir)
				if isinstance(name,int):
					data_matrix = np.array([contraste,rt,performance,confidence,selected_side,
										orientation,phase,name*np.ones_like(rt),session*np.ones_like(rt)]).squeeze().T
				else:
					data_matrix = np.array([contraste,rt,performance,confidence,selected_side,
										orientation,phase,session*np.ones_like(rt)]).squeeze().T
				yield data_matrix
		elif self.experiment=='Auditivo':
			if self._single_session:
				data_files = [f for f in self.list_data_files(override_raw_data_dir) if int(re.search('(?<=sesion)[0-9]+',f).group())==self.session and not f.endswith('quest.mat') and f.endswith('.mat')]
			else:
				data_files = [f for f in self.list_data_files(override_raw_data_dir) if int(re.search('(?<=sesion)[0-9]+',f).group()) in self.session and not f.endswith('quest.mat') and f.endswith('.mat')]
			sessions = [int(re.search('(?<=sesion)[0-9]+',f).group()) for f in data_files]
			for f,session in zip(data_files,sessions):
				aux = io.loadmat(f)
				contraste = aux['QQ']
				rt = aux['RT']
				performance = aux['correct']
				confidence = aux['SEGU']+0.5
				confidence[confidence>1.] = 1.
				confidence[confidence<0.] = 0.
				selected_side = aux['RTA']
				confidence_rt = aux['SEGUTIME']
				target_location = aux['orden']
				name = self.get_name_from_data_dir(os.path.dirname(f),override_raw_data_dir)
				if isinstance(name,int):
					data_matrix = np.array([contraste,rt,performance,confidence,selected_side,
										confidence_rt,target_location,name*np.ones_like(rt),session*np.ones_like(rt)]).squeeze().T
				else:
					data_matrix = np.array([contraste,rt,performance,confidence,selected_side,
										confidence_rt,target_location,session*np.ones_like(rt)]).squeeze().T
				yield data_matrix
	
	def load_data(self,override_raw_data_dir=None):
		"""
		self.load_data(override_raw_data_dir=None)
		
		Iterates all the data using a comprehention list of
		self.iter_data calls. Returns a 2D numpy.ndarray where the
		axis=0 corresponds to separate trials and axis=1 is described
		by self.column_description
		
		Input:
			override_raw_data_dir: Refer to iter_data or list_data_files
				for a detailed description of the functionality of this
				input parameter.
		
		Output: 2D numpy.ndarray
		
		"""
		first_element = True
		for data_matrix in self.iter_data(override_raw_data_dir=override_raw_data_dir):
			if first_element:
				all_data = data_matrix
				first_element = False
			else:
				all_data = np.concatenate((all_data,data_matrix),axis=0)
		return all_data
	
	def column_description(self):
		"""
		self.column_description()
		
		Returns a list with the description of the data contained in each
		column of the load_data
		"""
		numeric_name = isinstance(self.name,int)
		if self.experiment=='Luminancia':
			if numeric_name:
				return ['mean target lum [cd/m^2]','RT [s]','performance','confidence','selected side','name','session']
			else:
				return ['mean target lum [cd/m^2]','RT [s]','performance','confidence','selected side','session']
		elif self.experiment=='2AFC':
			if numeric_name:
				return ['contraste','RT [s]','performance','confidence','selected side','orientation [º]','phase','name','session']
			else:
				return ['contraste','RT [s]','performance','confidence','selected side','orientation [º]','phase','session']
		elif self.experiment=='Auditivo':
			if numeric_name:
				return ['contraste','RT [s]','performance','confidence','selected side','confidence RT [s]','target location','name','session']
			else:
				return ['contraste','RT [s]','performance','confidence','selected side','confidence RT [s]','target location','session']
		else:
			raise ValueError('No column description available for the experiment: {0}'.format(self.experiment))
	
	def __getstate__(self):
		return {'name':self.name,'session':self.session,'experiment':self.experiment,'data_dir':self.data_dir}
	
	def __setstate__(self,state):
		self.__init__(name=state['name'],session=state['session'],experiment=state['experiment'],data_dir=state['data_dir'])

def unique_subject_sessions(raw_data_dir,filter_by_experiment=None,filter_by_session=None):
	"""
	subjects = unique_subjects(raw_data_dir,filter_by_experiment=None,filter_by_session=None)
	
	This function explores de data_dir supplied by the user and finds the
	unique subjects that participated in the experiment. The output is a
	list of anonimized subjectSession objects.
	
	"""
	package_logger.debug('Getting list of unique SubjectSession instances')
	package_logger.debug('Input arg filter_by_experiment = {0}'.format(filter_by_experiment))
	package_logger.debug('Input arg filter_by_session = {0}'.format(filter_by_session))
	must_disable_and_reenable_logging = package_logger.isEnabledFor('DEBUG')
	output = []
	experiments = [d for d in os.listdir(raw_data_dir) if os.path.isdir(os.path.join(raw_data_dir,d))]
	for experiment in experiments:
		# We want the data of all experiments except sperling's
		if experiment=='sperling':
			continue
		if filter_by_experiment:
			if experiment!=filter_by_experiment:
				continue
		
		experiment_data_dir = os.path.join(raw_data_dir,experiment,'COMPLETOS')
		subject_rel_dirs = sorted([d for d in os.listdir(experiment_data_dir) if os.path.isdir(os.path.join(experiment_data_dir,d))])
		for subject_rel_dir in subject_rel_dirs:
			name = subject_rel_dir.lower()
			subject_dir = os.path.join(experiment_data_dir,subject_rel_dir)
			files = os.listdir(subject_dir)
			if experiment!='Luminancia':
				sessions = sorted(list(set([int(re.search('(?<=sesion)[0-9]+',f).group()) for f in files])))
			else:
				blocks = sorted(list(set([int(re.search('(?<=_B)[0-9]+(?=_)',f).group()) for f in files])))
				sessions = sorted(list(set([(block-1)//4+1 for block in blocks])))
			for session in sessions:
				if filter_by_session:
					if session!=filter_by_session:
						continue
				if must_disable_and_reenable_logging:
					logging.disable('DEBUG')
				output.append(SubjectSession(name,session,experiment,subject_dir))
				if must_disable_and_reenable_logging:
					logging.disable(logging.NOTSET)
	return anonimize_subjects(output)

def anonimize_subjects(subjectSessions):
	"""
	subjectSessions = anonimize_subjects(subjectSessions)
	
	Takes a list of SubjectSession objects and converts their names into
	a numerical id that overrides their original names.
	
	"""
	package_logger.debug('Anonimizing {0} SubjectSession instances'.format(len(subjectSessions)))
	names = []
	for ss in subjectSessions:
		if ss._single_data_dir:
			names.append(ss.name)
		else:
			names.extend(ss.name)
	names = sorted(list(set(names)))
	name_to_id = {}
	for subject_id,name in enumerate(names):
		name_to_id[name] = subject_id
	for ss in subjectSessions:
		if ss._single_data_dir:
			ss.change_name(name_to_id[ss.name])
		else:
			for name in ss.name:
				ss.change_name(name_to_id[ss.name],ss.name)
	return subjectSessions

def filter_subjects_list(subjectSessions,criteria='all_experiments'):
	"""
	filter_subjects_list(subjectSessions,criteria='all_experiments')
	
	Take a list of subjectSession instances and filter it according to
	the supplied criteria. Available criterias are:
	'all_experiments': Remove the subjects that did not perform all of
		the experiments performed by the rest of the subjects in the
		supplied list.
	'all_sessions_by_experiment': Remove the subjects that did not
		perform the maximum number of sessions for all experiments
	'experiment_{experiment_name}': Remove all the subjectSessions with
		an experiment not named as the supplied {experiment_name}.
	
	Output: A list of subjectSessions that satisfy the criteria.
	
	"""
	package_logger.debug('Filtering list of SubjectSession instances using criteria: {0}'.format(criteria))
	criteria = str(criteria)
	output = []
	if criteria=='all_experiments':
		names = [s.get_name() for s in subjectSessions]
		experiments = [s.experiment for s in subjectSessions]
		unique_names = sorted(list(set(names)))
		n_experiments = len(set(experiments))
		for name in unique_names:
			if len(set([e for n,e in zip(names,experiments) if n==name]))==n_experiments:
				output.extend([s for s in subjectSessions if s.get_name()==name])
	elif criteria=='all_sessions_by_experiment':
		names = [s.get_name() for s in subjectSessions]
		experiments = [s.experiment for s in subjectSessions]
		sessions = [s.session for s in subjectSessions]
		unique_names = sorted(list(set(names)))
		unique_experiments = sorted(list(set(experiments)))
		n_sessions = [len(set([s for s,e in zip(sessions,experiments) if e==ue])) for ue in unique_experiments]
		for name in unique_names:
			satifies_filter = True
			for i,experiment in enumerate(unique_experiments):
				if len(set([s for s,n,e in zip(sessions,names,experiments) if n==name and e==experiment]))!=n_sessions[i]:
					satifies_filter = False
					break
			if satifies_filter:
				output.extend([s for s in subjectSessions if s.get_name()==name])
	elif criteria.startswith('experiment'):
		experiment = criteria.split('_')[1]
		output = [s for s in subjectSessions if s.experiment==experiment]
	else:
		raise ValueError('The specified criteria: "{0}" is not implemented'.format(criteria))
	package_logger.debug('Filter input length {0}. Output length {1}'.format(len(subjectSessions),len(output)))
	return output

def merge_data_by_experiment(subjectSessions,filter_by_experiment=None,filter_by_session=None,return_column_headers=False):
	unique_experiments = sorted(list(set([s.experiment for s in subjectSessions])))
	output = {}
	if return_column_headers:
		output['headers'] = {}
	for experiment in unique_experiments:
		if filter_by_experiment:
			if experiment!=filter_by_experiment:
				continue
		output[experiment] = None
		if return_column_headers:
			output['headers'][experiment] = None
		merged_data = None
		for s in (s for s in subjectSessions if s.experiment==experiment):
			if filter_by_session:
				if s.session!=filter_by_session:
					continue
			data = s.load_data()
			if merged_data is None:
				merged_data = data
			else:
				merged_data = np.vstack((merged_data,data))
			if return_column_headers:
				if output['headers'][experiment] is None:
					output['headers'][experiment] = s.column_description()
		output[experiment] = merged_data
	return output

def merge_subjectSessions(subjectSessions,merge='all'):
	"""
	merge_subjectSessions(subjectSessions,merge='all')
	
	Take a list of SubjectSession instances and merge them according to
	the criteria specified in input merge. Available merge values:
	'all': All the SubjectSession instances with matching experiments
		are merged into a single SubjectSession instance.
	'sessions': All the SubjectSession instances with matching experiments
		and sessions are merged into a single SubjectSession instance.
	'names': All the SubjectSession instances with matching experiments
		and names are merged into a single SubjectSession instance.
	
	Output: a list of SubjectSession instances.
	
	"""
	package_logger.debug('Merging SubjectSession instances with merge method: {0}'.format(merge))
	merge = merge.lower()
	if merge=='all':
		names = {}
		data_dirs = {}
		sessions = {}
		for ss in subjectSessions:
			exp = ss.experiment
			if exp not in data_dirs.keys():
				data_dirs[exp] = []
			if exp not in names.keys():
				names[exp] = []
			if ss._single_data_dir:
				names[exp].append(ss.name)
				data_dirs[exp].append(ss.data_dir)
			else:
				names[exp].extend(ss.name)
				data_dirs[exp].extend(ss.data_dir)
			if exp not in sessions.keys():
				sessions[exp] = []
			if ss._single_session:
				sessions[exp].append(ss.session)
			else:
				sessions[exp].extend(ss.session)
		output = [SubjectSession(names[exp],sessions[exp],exp,data_dirs[exp]) for exp in data_dirs.keys()]
	elif merge=='sessions':
		sessions = {}
		for ss in subjectSessions:
			exp = str(ss.experiment)
			name = ss.name
			data_dir = ss.data_dir
			key = exp+'_'+ss.get_name()
			if key not in sessions.keys():
				sessions[key] = {'data':[],'name':name,'experiment':exp,'data_dir':data_dir}
			if ss._single_session:
				sessions[key]['data'].append(ss.session)
			else:
				sessions[key]['data'].extend(ss.session)
		output = [SubjectSession(sessions[key]['name'],sessions[key]['data'],sessions[key]['experiment'],sessions[key]['data_dir']) for key in sessions.keys()]
	elif merge=='names':
		data_dirs = {}
		for ss in subjectSessions:
			exp = str(ss.experiment)
			session = ss.session
			data_dir = ss.data_dir
			key = exp+'_'+ss.get_session()
			if key not in data_dirs.keys():
				data_dirs[key] = {'data':[],'name':[],'session':session,'experiment':exp}
			if ss._single_data_dir:
				data_dirs[key]['data'].append(ss.data_dir)
				data_dirs[key]['name'].append(ss.name)
			else:
				data_dirs[key]['data'].extend(ss.data_dir)
				data_dirs[key]['name'].extend(ss.name)
		output = [SubjectSession(data_dirs[key]['name'],data_dirs[key]['session'],data_dirs[key]['experiment'],data_dirs[key]['data']) for key in data_dirs.keys()]
	else:
		ValueError('Unknown merge criteria "{0}"'.format(merge))
	package_logger.debug('Merge results. Input length: {0} --- Output length: {1}'.format(len(subjectSessions),len(output)))
	return output

def increase_histogram_count(d,n):
	"""
	(out,indexes)=increase_histogram_count(d,n)
	
	Take an numpy.array of data d of shape (m,) and return an array out
	of shape (n,) that copies the elements of d keeping d's histogram
	approximately invariant. Second output "indexes" is an array so
	that out = d(indexes)
	
	"""
	d = d.squeeze()
	if len(d.shape)>1:
		raise(ValueError('Input data must be an array with only one dimension'))
	if n<len(d):
		raise(ValueError('n must be larger than the length of the data'))
	ud, ui, histogram = np.unique(d, return_inverse=True, return_counts=True)
	increased_histogram = np.floor(histogram*n/len(d))
	
	if np.sum(increased_histogram)<n:
		temp = np.zeros_like(histogram)
		cumprob = np.cumsum(histogram)/sum(histogram)
		for i in range(int(n-sum(increased_histogram))):
			ind = np.searchsorted(cumprob,random.random(),'left')
			temp[ind]+=1
		increased_histogram+=temp
	
	unique_indexes = []
	for i in range(len(ud)):
		unique_indexes.append(np.random.permutation([j for j,uii in enumerate(ui) if uii==i]))
	
	out = np.zeros(n)
	indexes = np.zeros_like(out,dtype=np.int)
	count_per_value = np.zeros_like(increased_histogram)
	for c in range(n):
		cumprob = np.cumsum(increased_histogram-count_per_value)/sum(increased_histogram-count_per_value)
		ind = np.searchsorted(cumprob,random.random(),'left')
		out[c] = ud[ind]
		indexes[c] = np.int(unique_indexes[ind][count_per_value[ind]%histogram[ind]])
		count_per_value[ind]+=1
	randperm_indexes = np.random.permutation(n)
	return out[randperm_indexes], indexes[randperm_indexes]

def compute_roc(performance,confidence,partition=101):
	"""
	compute_roc(performance,confidence,partition=101)
	
	Compute the Receiver Operation Characteristic (ROC) curve from two
	input arrays that hold performance (0 miss, 1 hit) and confidence.
	
	Input:
		performance: 1D numpy array of binary performances (0 miss,
			1 hit)
		confidence: 1D numpy array with the same number of elements as
			performance that holds confidence floating point values in
			the range [0,1].
		partition: Int that represents the number of edges that will be
			used to partition the [0,1] confidence interval to compute
			the ROC curves
	
	Output:
		roc: 2D numpy array whos shape is (partition,2).
			roc[:,0] = P(confidence<x|hit)
			roc[:,1] = P(confidence<x|miss)
	
	"""
	edges = np.linspace(0,1,int(partition))
	roc = np.zeros((int(partition),2),dtype=np.float)
	hit = performance==1
	miss = np.logical_not(hit)
	nhits = np.sum(hit.astype(np.float))
	nmisses = np.sum(miss.astype(np.float))
	for i,ue in enumerate(edges):
		if i<int(partition)-1:
			p1 = np.sum(np.logical_and(confidence<ue,hit).astype(np.float))/nhits
			p2 = np.sum(np.logical_and(confidence<ue,miss).astype(np.float))/nmisses
		else:
			p1 = np.sum(np.logical_and(confidence<=ue,hit).astype(np.float))/nhits
			p2 = np.sum(np.logical_and(confidence<=ue,miss).astype(np.float))/nmisses
		roc[i,:] = np.array([p1,p2])
	return roc

def compute_auc(roc):
	"""
	compute_auc(roc)
	
	Compute the area under the ROC curve.
	Input:
		roc: A 2D numpy array as the one returned by compute_roc.
	
	Output:
		auc: A float that is the area under the ROC curve
	
	"""
	return scipy.integrate.trapz(roc[:,1],roc[:,0])

def test(raw_data_dir='/home/luciano/Dropbox/Luciano/datos joaquin/para_luciano/raw_data'):
	try:
		from matplotlib import pyplot as plt
		loaded_plot_libs = True
	except:
		loaded_plot_libs = False
	subjects = unique_subject_sessions(raw_data_dir)
	try:
		subjects = unique_subject_sessions(raw_data_dir)
	except:
		raw_data_dir = raw_data_dir.replace('/home/','/Users/')
		subjects = unique_subject_sessions(raw_data_dir)
	
	#~ bla = {'2AFC':[],'Auditivo':[],'Luminancia':[]}
	#~ for s in subjects:
		#~ rt = np.sort(s.load_data()[:,1])
		#~ key = s.experiment
		#~ bla[key].append(np.sum((rt<8.).astype(np.float))/float(len(rt)))
	
	print(str(len(subjects))+' subjectSessions can be constructed found')
	filtered_subjects = filter_subjects_list(subjects)
	print(str(len(filtered_subjects))+' filtered subjectSessions with all_experiments criteria')
	subjects = filter_subjects_list(subjects,'all_sessions_by_experiment')
	print(str(len(subjects))+' filtered subjectSessions with sessions_by_experiment criteria')
	
	merged_all = merge_subjectSessions(subjects,merge='all')
	print(str(len(merged_all))+' merged subjectSessions with merge all')
	merged_sessions = merge_subjectSessions(subjects,merge='sessions')
	print(str(len(merged_sessions))+' merged subjectSessions with merge sessions')
	merged_subjects = merge_subjectSessions(subjects,merge='names')
	print(str(len(merged_subjects))+' merged subjectSessions with merge subjects')
	
	experiments_data = merge_data_by_experiment(subjects,return_column_headers=True)
	
	print('Successfully merged all subjects data in '+str(len([k for k in experiments_data.keys() if k!='headers']))+' experiments')
	headers = experiments_data['headers']
	for key in experiments_data.keys():
		if key=='headers':
			continue
		data = experiments_data[key]
		#~ data[:,0] = np.round(data[:,0]*5e3)/5e3
		print(key,len(set(data[:,0])))
		matches = True
		for test_subj in [t for t in merged_all if t.experiment==key]:
			testdata = test_subj.load_data()
			test = testdata.shape[0]==data.shape[0]
			if not test:
				print(testdata.shape,data.shape)
			matches = matches and test
		print('Merged all matches shape? {0}'.format('Yes' if matches else 'No'))
		matches = True
		for test_subj in [t for t in merged_sessions if t.experiment==key]:
			testdata = test_subj.load_data()
			test = testdata.shape[0]==data[data[:,-2]==int(test_subj.name)].shape[0]
			if not test:
				print(test_subj.name,testdata.shape,data[data[:,-2]==test_subj.name].shape)
			matches = matches and test
		print('Merged all sessions matches shape? {0}'.format('Yes' if matches else 'No'))
		matches = True
		for test_subj in [t for t in merged_subjects if t.experiment==key]:
			testdata = test_subj.load_data()
			test = testdata.shape[0]==data[data[:,-1]==test_subj.session].shape[0]
			if not test:
				print(testdata.shape,data[data[:,-1]==test_subj.session].shape)
			matches = matches and test
		print('Merged all subjects matches shape? {0}'.format('Yes' if matches else 'No'))
		print('{0}: {1} trials, {2} sessions, {3} subjects'.format(key,data.shape[0],len(np.unique(data[:,-1])),len(np.unique(data[:,-2]))))
		if loaded_plot_libs:
			inds = data[:,1]<14.
			plt.figure()
			plt.subplot(141)
			plt.hist(data[inds,0],100,normed=True)
			plt.xlabel(headers[key][0])
			plt.subplot(142)
			plt.hist(data[inds,1],100,normed=True)
			plt.xlabel(headers[key][1])
			plt.subplot(143)
			plt.hist(data[inds,2],2,normed=True)
			plt.xlabel(headers[key][2])
			plt.subplot(144)
			plt.hist(data[inds,3],100,normed=True)
			plt.xlabel(headers[key][3])
			plt.suptitle(key)
			
			roc = compute_roc(data[inds,2],data[inds,3])
			auc = compute_auc(roc)
			plt.figure()
			plt.plot(roc[:,0],roc[:,1])
			plt.xlabel(r'$P(conf<x|hit)$')
			plt.ylabel(r'$P(conf<x|miss)$')
			plt.title(key+' (AUC = {0})'.format(auc))
	plt.show(True)

if __name__=="__main__":
	if len(sys.argv)>1:
		test(sys.argv[1])
	else:
		test()
