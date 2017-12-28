# Mathematics and Plotting
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as sstats
from scipy.stats import gaussian_kde as gkde
plt.rcParams.update({'font.size': 14})




def sandbox(num_samples = int(1E4), lam_bound = [3,6], lam_0=3.5, 
t_0 = 0.1, Delta_t = 0.1, num_observations = 4, sd=1, 
fixed_noise = True, compare = False, smooth_post = False, num_trials = 1):
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
        
    # Create observations... additive noise.
    obs_data = lam_0 * np.exp(-t) + np.random.randn(int(num_observations))*sigma
    
    # Global options - Consistent over all the trials
    plt.rcParams['figure.figsize'] = (18, 6)
    plt.close('all')
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    trial_seeds = [trial for trial in range(num_trials)] # seed each trial in the same numerical order
    entropy_list = []
    num_accept_list = []
    
    for seed in trial_seeds:
        if not fixed_noise:
            obs_data = lam_0 * np.exp(-t) + np.random.randn(int(num_observations))*sigma
            
        np.random.seed(seed)
        # Sample the Parameter Space
        a, b = lam_bound
        lam = np.random.uniform(a, b, size = (int(num_samples), 1) ) # standard uniform

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
        if compare or smooth_post:
            if seed == 0:
                print('Performing Accept/Reject to estimate the pushforward of the posterior.')
            accept_inds = [i for i in range(num_samples) if eta_r[i] > np.random.uniform(0,1) ] 
            num_accept = len(accept_inds)
            num_accept_list.append(num_accept)
        
        entropy_list.append( sstats.entropy( obs_dens.pdf(D), pf_dens.evaluate(D) ) )    
        
        res = 50;
        max_x = D.max();
        # Plot stuff
        # plt.figure(1)
        x1 = np.linspace(-0.25, max_x, res)
        ax1.plot(x1, pf_dens.evaluate(x1))
        plt.title('Pushforward Q(Prior)')
        plt.xlabel('Q(lambda)')
        
        x2 = np.linspace(0, max_x, res)
        ax2.plot(x2, obs_dens.pdf(x2))
        if compare:
            push_post_dens_kde = gkde(D[accept_inds])
            pf = push_post_dens_kde.pdf(x2)
            ax2.plot(x2, pf/np.sum(pf))
            plt.legend(['Observed','Recovered'])
        plt.title('Observed Density')
        plt.xlabel('Q(lambda)')
        

        x3 = np.linspace(a,b, res)
        if smooth_post:
            input_dim = lam.shape[1] # get input dimension by observing the shape of lambda
            post_dens_kde = gkde(np.array([lam[accept_inds, i] for i in range(input_dim)]))
            ps = post_dens_kde.pdf(x3)
            ax3.plot(x3, ps)
            
        else:    
            ax3.scatter(lam, eta_r)
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
    
    print('\tMean Entropy is: %.2f with var %.2f'%(np.mean(entropy_list), np.std(entropy_list)))
    print('Entropies: ')
    if compare or smooth_post:
        print(['%.2f '%entropy_list[n] for n in range(num_trials)])
        print('Median Acceptance Rate: %2.2f%%'%(np.mean(num_accept_list)/num_samples) )
    #     return eta_r


