#!/usr/bin/python
## Author: Michael Pilosov
## Copyright 2017
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sstats 
from scipy.stats import gaussian_kde as gauss_kde

def comparepush(x, obs_dens, post_dens):
    plt.plot(x, obs_dens.pdf(x.transpose()), 'y', label='obs')
    plt.plot(x, post_dens.evaluate(x), 'c', label='$O(post)')
    plt.xlabel('$O(\lambda)$')
    plt.legend()
    plt.show()

def compare_input_dens(x, analytical_dens, estimated_dens, viewdim=0, lab_1='prior', lab_2='KDE prior', title=''):
    # specify viewdim (list) to view crosssections e.g. [0,1] gives you the diagonal view through the first two dimensions
    input_dim = estimated_dens.d
    num_samples = len(x)
    y = np.zeros( (input_dim, num_samples) )
    y[viewdim,:] = x 
    plt.plot(x, analytical_dens.pdf( y.transpose() ), 'y', label=lab_1)
    plt.plot(x, estimated_dens.evaluate(y), 'c', label=lab_2)
    if type(viewdim) == int:
        plt.xlabel('$\lambda_%d$'%viewdim)
    else:
        plt.xlabel('$\lambda_{%s}$'%str(viewdim))
    plt.title(title)
    plt.legend()
    plt.show()
    
def compare_output_dens(x, analytical_dens, estimated_dens, viewdim=0, lab_1='observed', lab_2='KDE push', title=''):
    # specify viewdim (list) to view crosssections e.g. [0,1] gives you the diagonal view through the first two dimensions
    dim = estimated_dens.d
    num_samples = len(x)
    y = np.zeros( (num_samples, dim) )
    y[:,viewdim] = x 
    d1 = analytical_dens.pdf(y)[:,viewdim]
    d2 = estimated_dens.pdf(y)
    plt.plot(x, d1/np.max(d1), 'y', label=lab_1)
    plt.plot(x, d2/np.max(d2), 'c', label=lab_2)
    
    if type(viewdim) == int:
        plt.xlabel('$O(\lambda_%d)$'%viewdim)
    else:
        plt.xlabel('$O(\lambda_{%s})$'%str(viewdim))
    plt.title(title)
    plt.legend()
    # plt.show()

def view_analytical_dens(x, analytical_dens, viewdim=0, title=''):
    try:
        dim = analytical_dens.dim # sstats multivariate normal in case someone wants covariance.
    except AttributeError:
        test_point = analytical_dens.rvs()
        try:
            dim = len(test_point)
        except TypeError:
            try:
                dim = test_point.shape[0]
            except IndexError:
                print(Warning("One-dimensional density inferred. Make sure this is what you wanted."))
                dim = 1
    num_samples = len(x)
    y = np.zeros( (num_samples, dim) )
    y[:,viewdim] = x # specify dim (list) to view crosssections e.g. [0,1] gives you the diagonal view through the first two dimensions
    d = analytical_dens.pdf(y)[:,viewdim]
    plt.plot(x, d, 'y')
    
    # plt.plot(x, analytical_dens.pdf(y)[:,viewdim], 'y', label="dim=%d"%viewdim)
    if type(viewdim)==int:
        plt.xlabel('$x_%d$'%viewdim)
    # else:
        # plt.xlabel('$\lambda_{%s}$'%str(viewdim))
    plt.title(title)
    # plt.legend()
    plt.show()
 

def view_est_dens(x, estimated_dens, viewdim=0, lab='KDE', title=''):
    dim = estimated_dens.d
    num_samples = len(x)
    y = np.zeros( (num_samples, dim) )
    y[:,viewdim] = x # specify dim (list) to view crosssections e.g. [0,1] gives you the diagonal view through the first two dimensions
    plt.plot(x, estimated_dens.pdf(y), 'y', label=lab)
    if type(viewdim)==int:
        plt.xlabel('$x_%d$'%viewdim)
    # else:
        # plt.xlabel('$x_{%s}$'%str(viewdim))
    plt.title(title)
    plt.legend()
    plt.show()
 
def compare_est_input_dens(x, estimated_dens1, estimated_dens2, viewdim=0, lab_1='KDE prior', lab_2='KDE post', title=''):
    input_dim = estimated_dens1.d
    num_samples = len(x)
    y = np.zeros( (num_samples, input_dim) )
    y[:, viewdim] = x # specify dim (list) to view crosssections e.g. [0,1] gives you the diagonal view through the first two dimensions
    d1 = estimated_dens1.pdf(y)
    d2 = estimated_dens2.pdf(y)
    plt.plot(x, d1/np.max(d1), 'y', label=lab_1)
    plt.plot(x, d2/np.max(d2), 'c', label=lab_2)
    if type(viewdim)==int:
        plt.xlabel('$\lambda_%d$'%viewdim)
    else:
        plt.xlabel('$\lambda_{%s}$'%str(viewdim))
    plt.title(title)
    plt.legend()
    plt.show()
    
def pltaccept(lam, inds, N, eta_r, i=0, j=1): # plots first N of accepted, any 2D marginals specified
    lam_accept = lam[inds,:]
    lam_min, lam_max = np.min(lam), np.max(lam)
    if i == j:
#         inds = [k for k in range(N+1) if lam[k, i] in lam_accept]
        plt.scatter(lam[:, i], eta_r[:])
        plt.ylabel('$\eta$')
        plt.xlabel('$\lambda_%d$'%i)
        plt.xlim([lam_min, lam_max])
        plt.scatter(lam_accept[0:N,i], eta_r[inds[0:N]])
    else:
#         plt.scatter(lam[:, i], lam[:, j], s=1)
        plt.scatter(lam_accept[0:N, i], lam_accept[0:N, j], s=4)
        plt.xlabel('$\lambda_%d$'%i)
        plt.ylabel('$\lambda_%d$'%j)
        plt.xlim([lam_min, lam_max])
        plt.ylim([lam_min, lam_max])
    plt.show()
