# Mathematics and Plotting
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as sstats
from scipy.stats import gaussian_kde as gkde
plt.rcParams.update({'font.size': 14})
plt.rcParams['figure.figsize'] = 5, 5



def sandbox(num_samples = int(1E4), lam_bound = [3,6], lam_0=3.5, 
            t_0 = 0.1, Delta_t = 0.1, num_observations = 4, sd=1):
    # NOTE this version only uses constant variances for the sake
    # of interactivity.
    # TODO overload sd variable to take in lists/arrays
    sigma = sd*np.ones(num_observations)
    
    if num_observations == 1:
        print('K=0 specified, This is a single observation at t = %f.'%t_0)
        
    t = np.linspace(t_0, t_0 + Delta_t*(num_observations-1), num_observations)
    
    def Q_fun(lam, obs_data):
        predictions = lam*np.exp(-t)
        residuals = predictions - obs_data
        QoI = np.sum( (residuals/sigma)**2 )
        return QoI
    
    # Sample the Parameter Space
    a, b = lam_bound
    lam = np.random.uniform(a, b, size = (int(num_samples), 1) ) # standard uniform
    
    # Create observations
    obs_data = lam_0 * np.exp(-t) + np.random.randn(int(num_observations))*sigma
    
    # Map to Data Space
    D = np.zeros(int(num_samples))
    for i in range(int(num_samples)):
        D[i] = Q_fun(lam[i,:], obs_data)
    
#     print('dimensions :  lambda = ' + str(lam.shape) + '   D = ' + str(D.shape) )
    # Perform KDE to estimate the pushforward
    pf_dens = gkde(D) # compute KDE estimate of it
    # Specify Observed Measure - Uniform Density
    
    #obs_dens = sstats.uniform(0,uncertainty) # 1D only
    obs_dens = sstats.chi2(int(num_observations))
    
    # Solve the problem
    r = obs_dens.pdf(D) / pf_dens.evaluate(D) # vector of ratios evaluated at all the O(lambda)'s
    M = np.max(r)

    r = r[:,np.newaxis]
    eta_r = r[:,0]/M
    
    print('\tEntropy is %1.4e'%sstats.entropy( obs_dens.pdf(D), pf_dens.evaluate(D) ))
    
    res = 50;
    max_x = D.max();
    # Plot stuff
    plt.rcParams['figure.figsize'] = (18, 6)
    plt.figure()
    plt.subplot(1, 3, 1)
    x = np.linspace(-0.25, max_x, res)
    plt.plot(x, pf_dens.evaluate(x))
    plt.title('Pushforward of Prior')
    plt.xlabel('O(lambda)')
    
    plt.subplot(1, 3, 2)
    xx = np.linspace(0, max_x, res)
    plt.plot(xx, obs_dens.pdf(xx))
    plt.title('Observed Density')
    plt.xlabel('O(lambda)')

    plt.subplot(1, 3, 3)
    plt.scatter(lam, eta_r)
    # plt.plot(lam_accept, gkde(lam_accept))
    plt.scatter(lam_0, 0.05)
    plt.title('Posterior Distribution') #\nof Uniform Observed Density \nwith bound = %1.2e'%uncertainty)
    plt.xlabel('Lambda')
#     plt.title('$\eta_r$')
    # # OPTIONAL:
    # pr = 0.2 # percentage view-window around true parameter.
#     plt.xlim(lam0*np.array([1-pr,1+pr]))
    plt.xlim([a,b])
    plt.show()
    
#     return eta_r


