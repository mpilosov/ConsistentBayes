#!/usr/bin/python
## Author: Michael Pilosov
## Copyright 2017
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import scipy.stats as sstats 
from scipy.stats import gaussian_kde as gauss_kde


def pltdata(data, view_dim_1=0, view_dim_2=1, eta_r=None, inds=None, N=None,  color="eggplant", space=0.05, svd=False): # plots first N of accepted, any 2D marginals specified
    if type(data) is np.ndarray:

        if inds is not None:
            data_subset = data[inds,:]
        else:
            data_subset = data
        if N is not None:
            data_subset = data_subset[0:N]
    else:
        try: # try to infer the dimension... 
            d = len(data.rvs())
        except TypeError:
            try:
                d = data.rvs().shape[1]
            except IndexError:
                d = 1
         
        data = data.rvs((N,d)) # if we get a distribution object, use it to generate samples.  
        data_subset = data 
    x_data = data_subset[:, view_dim_1]
    try:
        y_data = data_subset[:, view_dim_2]
    except IndexError:
        y_data = x_data
    rgb_color = sb.xkcd_rgb[color]

    if view_dim_1 == view_dim_2:
        sb.kdeplot(x_data, color=rgb_color)
        if eta_r is not None:
            plt.figure()
            plt.scatter(data[:,view_dim_1], eta_r, alpha=0.1, color=rgb_color)
    else:
            # perform SVD and show secondary plot
        if svd:
            offset = np.mean(data_subset, axis=0)
            la = data_subset - np.array(offset)
            U,S,V = np.linalg.svd(la)
            new_data = np.dot(V, la.transpose()).transpose() + offset
            x_data_svd = new_data[:,view_dim_1]
            y_data_svd = new_data[:,view_dim_2]
            
            sb.jointplot(x=x_data_svd, y=y_data_svd, kind='kde', 
                         color=rgb_color, space=space, stat_func=None)
        
        else: # no SVD - show scatter plot
            plt.figure()
            if inds is None:
                plt.scatter(data[0:N,view_dim_1], data[0:N,view_dim_2], alpha=0.2, color=rgb_color)
                
            else:
                plt.scatter(data[inds[0:N],view_dim_1], data[inds[0:N],view_dim_2], alpha=0.2, color=rgb_color)
                # plt.axis('equal')
                min_1 = np.min(data[:,view_dim_1])
                min_2 = np.min(data[:,view_dim_2])
                max_1 = np.max(data[:,view_dim_1])
                max_2 = np.max(data[:,view_dim_2])
                plt.xlim([min_1, max_1])
                plt.ylim([min_2, max_2])
        sb.jointplot(x=x_data, y=y_data, kind='kde', 
                     color=rgb_color, space=space, stat_func=None)
        
    plt.show()


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



    
 

