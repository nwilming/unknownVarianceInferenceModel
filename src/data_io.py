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
import scipy.io
import os, itertools, sys, random, re, scipy.integrate, logging
from utils import parse_details_file

package_logger = logging.getLogger("data_io")
package_logger.debug('Parsing details file')
parsed_details_file = parse_details_file()
raw_data_dir = parsed_details_file['raw_data_dir']

class SubjectSession:
	def __init__(self,experiment,data_dir,name=None,session=None):
		"""
		SubjectSession(self,session,experiment,data_dir,name=None)
		
		This class is an interface to flexible load the data from
		Ais et al 2016 of all 3 2AFC tasks. Upon creation, a list of
		subject names, a list of sessions and a single experiment name
		is provided along with a list of directory paths where the
		corresponding data experimental data is located.
		This class can then be used to load the data of the different
		experiments into a standarized numpy array format.
		
		Input:
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
			'name': None, a string or a list of strings that represent
				subject names. These names can be arbitrary and
				unrelated to the real name of the subject. In fact, the
				package data_io provides a function to anonimize the
				subjects. If None, the subject's name is inferred from
				the directory structure.
		
		"""
		package_logger.debug('Creating SubjectSession instance')
		self.logger = logging.getLogger("data_io.SubjectSession")
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
		self.get_experiment_details()
		self.logger.debug("Instance's experiment: {0}".format(self.experiment))
		if isinstance(data_dir,list):
			if name is None:
				self.name = [str(d) for d in data_dir]
			else:
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
			if name is None:
				self.name = str(data_dir)
			else:
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
	
	def get_name_from_data_dir(self,data_dir):
		"""
		self.get_name_from_data_dir(data_dir)
		
		Returns the name string that corresponds to the supplied data_dir.
		
		"""
		return self._map_data_dir_name[data_dir]
	
	def get_experiment_details(self):
		if not self.experiment in parsed_details_file['experiment_details'].keys():
			raise ValueError('The desired experiment "{0}" is not declared in the experiment_details.txt file'.format(self.experiment))
		temp = parsed_details_file['experiment_details'][self.experiment]['IO']
		self.session_parser = temp['session_parser']
		self.file_extension = temp['file_extension']
		self.time_conversion_to_seconds = temp['time_conversion_to_seconds']
		try:
			self.excluded_files = temp['excluded_files']
		except:
			self.excluded_files = None
		self.data_structure = temp['data_structure']
		self.data_fields = self.data_structure['data_fields']
		if self.file_extension=='.mat':
			self.raw_data_loader = lambda f: scipy.io.loadmat(f)
		else:
			delimiter = self.data_structure['delimiter']
			self.raw_data_loader = lambda f: np.loadtxt(f, delimiter=delimiter, unpack=True)
	
	def change_name(self,new_name,orig=None):
		"""
		self.change_name(new_name,orig=None)
		
		Change the subjectSession name with the supplied new_name string.
		If the subjectSession has a list of names associated to it, it
		is mandatory to supply the original name which is going to be
		replaced in the input 'orig'.
		
		"""
		self.logger.debug('Changing name from {0} to {1}'.format(orig if not orig is None else self.name, new_name))
		if not self._single_data_dir:
			if orig is None:
				raise ValueError('Must supply the original name value when the SubjectSession has more than one data_dir')
			changed_index = self.name.index(orig)
			self.name[changed_index] = new_name
			self._map_data_dir_name[self.data_dir[changed_index]] = new_name
		else:
			self.name = new_name
			self._map_data_dir_name[self.data_dir] = new_name
	
	def list_data_files(self):
		"""
		self.list_data_files()
		
		Returns a list of the data files in the
		os.path.join(raw_data_dir,self.data_dir) paths.
		
		"""
		if self._single_data_dir:
			data_dir = os.path.join(raw_data_dir,self.data_dir)
			return [os.path.join(data_dir,f) for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir,f))]
		else:
			listdirs = []
			for dd in self.data_dir:
				data_dir = os.path.join(raw_data_dir,dd)
				listdirs.extend([os.path.join(data_dir,f) for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir,f))])
			return list(set(listdirs))
	
	def iter_data(self):
		"""
		self.iter_data()
		
		Iterate over the experimental data files listed in the data_dirs.
		
		Output:
			Each yielded value is a 2D numpy.ndarray. axis=0 represent
			different trials and axis=1 different data for each trial.
			Refer to self.column_description to get a description of
			the data contained at each index of axis=1.
		
		"""
		if self._single_session:
			if not self.excluded_files is None:
				data_files = (f for f in self.list_data_files() if self.session_parser(f)==self.session and f.endswith(self.file_extension) and not re.search(self.excluded_files,f))
			else:
				data_files = (f for f in self.list_data_files() if self.session_parser(f)==self.session and f.endswith(self.file_extension))
		else:
			if not self.excluded_files is None:
				data_files = (f for f in self.list_data_files() if self.session_parser(f) in self.session and f.endswith(self.file_extension) and not re.search(self.excluded_files,f))
			else:
				data_files = (f for f in self.list_data_files() if self.session_parser(f) in self.session and f.endswith(self.file_extension))
		for f in data_files:
			self.logger.debug('Loading data from file {0}'.format(f))
			name = self.get_name_from_data_dir(os.path.dirname(f).replace(raw_data_dir,''))
			session = self.session_parser(f)
			raw_data = self.raw_data_loader(f)
			
			mandatory_fields = ['contrast','rt','performance','confidence'] # variance is another important field if the fits are done with an unknown variance decision model
			fields = []
			for mf in mandatory_fields:
				if mf not in self.data_fields:
					raise RuntimeError('Field {0} must be specified in the experiment_details.txt for experiment {1}'.format(mf,self.experiment))
				fields.append(mf)
			for df in self.data_fields:
				if df in fields:
					continue
				fields.append(df)
			data_dict = {}
			for field in fields:
				data_dict[field] = np.squeeze(eval(self.data_fields[field])(raw_data))
			data_dict['rt']*=self.time_conversion_to_seconds
			data_dict['confidence'][data_dict['confidence']>1] = 1.
			data_dict['confidence'][data_dict['confidence']<0] = 0.
			l = len(data_dict[fields[0]])
			if any([len(data_dict[f])!=l for f in fields[1:]]):
				raise RuntimeError('Inconsistent size of the raw_data for file {0}. All fields must have the same number of elements'.format(f))
			
			dtype = [(str(fi),np.float) for fi in fields]
			dtype.extend([(str('name'),str('S32')),(str('session'),np.int),(str('experiment'),str('S32'))])
			for ind in range(l):
				trial_data = [data_dict[field][ind] for field in fields]
				trial_data.extend([name,session,self.experiment])
				yield np.array(tuple(trial_data),dtype=np.dtype(dtype))
	
	def load_data(self):
		"""
		self.load_data()
		
		Iterates all the data using a comprehention list of
		self.iter_data calls. Returns a numpy.ndarray where the
		axis=0 corresponds to separate trials and the element within
		is a structured array whose field names can be accesed by calling
		output[0].dtype.names
		
		Output: 1D numpy.ndarray where each element is itself a structured
			array
			Important field names:
				constrast: Signal strength
				rt: Response time in seconds
				performance: If the trial was successfully responded or not
				confidence: Responded confidence
				name: corresponding data's SubjectSession name
				session: corresponding data's SubjectSession session
				experiment: corresponding data's SubjectSession experiment
		
		"""
		return np.array([d for d in self.iter_data()])
	
	def __getstate__(self):
		return {'name':self.name,'session':self.session,'experiment':self.experiment,'data_dir':self.data_dir}
	
	def __setstate__(self,state):
		self.__init__(name=state['name'],session=state['session'],experiment=state['experiment'],data_dir=state['data_dir'])

def unique_subject_sessions(filter_by_experiment=None):
	"""
	subjects = unique_subjects(filter_by_experiment=None)
	
	This function explores the raw_data_dir specified in the
	experiment_details.txt file and finds the unique subjects that
	participated in the experiment. The output is a list of anonimized
	subjectSession instances.
	
	Input:
		filter_by_experiment: None or a list of valid experiment names
			that are to be stored in the output. If None, every
			experiment encountered is stored.
	
	"""
	package_logger.debug('Getting list of unique SubjectSession instances')
	package_logger.debug('Input arg filter_by_experiment = {0}'.format(filter_by_experiment))
	must_disable_and_reenable_logging = package_logger.isEnabledFor('DEBUG')
	output = []
	experiments = [d for d in os.listdir(raw_data_dir) if os.path.isdir(os.path.join(raw_data_dir,d))]
	for experiment in experiments:
		if not filter_by_experiment is None and experiment not in filter_by_experiment:
			continue
		session_parser = parsed_details_file['experiment_details'][experiment]['IO']['session_parser']
		for subject_rel_dir in sorted([d for d in os.listdir(os.path.join(raw_data_dir,experiment)) if os.path.isdir(os.path.join(raw_data_dir,experiment,d))]):
			name = subject_rel_dir.lower()
			subject_dir = os.path.join(experiment,subject_rel_dir)
			files = os.listdir(os.path.join(raw_data_dir,subject_dir))
			sessions = sorted(list(set([session_parser(f) for f in files])))
			for session in sessions:
				if must_disable_and_reenable_logging:
					logging.disable('DEBUG')
				output.append(SubjectSession(name=name,session=session,experiment=experiment,data_dir=subject_dir))
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
	package_logger.debug('Sorted list of unique names: {0}'.format(names))
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
		output = [SubjectSession(name=names[exp],session=sessions[exp],experiment=exp,data_dir=data_dirs[exp]) for exp in data_dirs.keys()]
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
		output = [SubjectSession(name=sessions[key]['name'],session=sessions[key]['data'],experiment=sessions[key]['experiment'],data_dir=sessions[key]['data_dir']) for key in sessions.keys()]
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
		output = [SubjectSession(name=data_dirs[key]['name'],session=data_dirs[key]['session'],experiment=data_dirs[key]['experiment'],data_dir=data_dirs[key]['data']) for key in data_dirs.keys()]
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

def _test():
	try:
		from matplotlib import pyplot as plt
		loaded_plot_libs = True
	except:
		loaded_plot_libs = False
	subjects = unique_subject_sessions()
	
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
	
	experiments_data = {me.experiment:me.load_data() for me in merged_all}
	
	print('Successfully merged all subjects data in '+str(len([k for k in experiments_data.keys()]))+' experiments')
	for key in experiments_data.keys():
		data = experiments_data[key]
		print(key,len(set(data['contrast'])))
		print('{0}: {1} trials, {2} sessions, {3} subjects'.format(key,data.shape[0],len(np.unique(data['session'])),len(np.unique(data['name']))))
		if loaded_plot_libs:
			inds = data['rt']<14.
			plt.figure()
			plt.subplot(141)
			plt.hist(data['rt'][inds],100,normed=True)
			plt.xlabel('rt')
			plt.subplot(142)
			plt.hist(data['contrast'][inds],100,normed=True)
			plt.xlabel('contrast')
			plt.subplot(143)
			plt.hist(data['performance'][inds],2,normed=True)
			plt.xlabel('performance')
			plt.subplot(144)
			plt.hist(data['confidence'][inds],100,normed=True)
			plt.xlabel('confidence')
			plt.suptitle(key)
			
			roc = compute_roc(data['performance'][inds],data['confidence'][inds])
			auc = compute_auc(roc)
			plt.figure()
			plt.plot(roc[:,0],roc[:,1])
			plt.xlabel(r'$P(conf<x|hit)$')
			plt.ylabel(r'$P(conf<x|miss)$')
			plt.title(key+' (AUC = {0})'.format(auc))
	plt.show(True)

if __name__=="__main__":
	_test()
