from numpy.distutils.core import setup, Extension

module1 = Extension('dp',
                    sources = ['dpmodule.cpp', 'DecisionPolicy.cpp'],
                    extra_compile_args = ['-Wno-write-strings','-O2'])

setup (name = 'DP',
       version = '1.0',
       description = 'Methods to compute the decision bounds for bayesian inference',
       ext_modules = [module1])
