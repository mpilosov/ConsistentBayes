# Mathematics and Plotting
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as sstats
from scipy.stats import gaussian_kde as gauss_kde
plt.rcParams.update({'font.size': 14})
from ipywidgets import widgets
import cbayes.sample as samp
import cbayes.solve as solve
import logging

def SSE_generator(model, obs_data, sigma=1):   # this generates a sum of squared residuals.
    def QoI_fun(inputs):         # that conforms to our desired model input
        predictions = model(inputs)
        assert predictions.shape[1] == len(obs_data)
        residuals = predictions - obs_data
        QoI = np.sum( (residuals/sigma)**2, axis=1 )
        return QoI
    return QoI_fun

def sandbox(num_samples = int(1E4), lam_bound = [3,6], lam_0=3.5, 
t_0 = 0.1, Delta_t = 0.1, num_observations = 4, sd=1, 
fixed_noise = True, compare = False, smooth_post = False, fun_choice = 0, num_trials = 1):
    # NOTE this version only uses constant variances for the sake
    # of interactivity.
    # TODO overload sd variable to take in lists/arrays
    np.random.seed(0) # want deterministic results
    sigma = sd*np.ones(num_observations)
    
    t = np.linspace(t_0, t_0 + Delta_t*(num_observations-1), num_observations)
    if fun_choice == 0:
        def model(lam):
            return lam*np.exp(-t)
    elif fun_choice == 1:
        def model(lam):
            return np.sin(lam*t)
    elif fun_choice == 2:
        def model(lam):
            return lam*np.sin(t)
    else:
        return None
        
    
    # Global options - Consistent over all the trials
    plt.rcParams['figure.figsize'] = (18, 6)
    plt.close('all')
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    trial_seeds = [trial for trial in range(num_trials)] # seed each trial in the same numerical order
    entropy_list = []
    num_accept_list = []
    # in the case that we fix our noise-model:
    observed_data = model(lam_0) + np.random.randn(int(num_observations))*sigma
    # Instantiate the sample set object.
    S = samp.sample_set(size=(num_samples,1))
    a, b = lam_bound
    S.set_dist('uniform',{'loc':a, 'scale':b-a}) # same distribution object for all
    
    for seed in trial_seeds:
        if not fixed_noise: # if we change the noise model from run-to-run, recompute observed_data
            np.random.seed(seed)
            observed_data = model(lam_0) + np.random.randn(int(num_observations))*sigma
        
        # np.random.seed(seed)
        # Sample the Parameter Space
        S.generate_samples(seed=seed)
        lam = S.samples
        QoI_fun = SSE_generator(model, observed_data, sigma) # generates a function that just takes `lam` as input
        P = samp.map_samples_and_create_problem(S, QoI_fun)
        # Map to Data Space
        D = P.output.samples.transpose()
        
        P.compute_pushforward_dist() # gaussian_kde by default on the data space.
        pf_dens = P.pushforward_dist
        
        P.set_observed_dist('chi2', {'df':num_observations}) # define your observed distribution.
        P.set_ratio() # compute ratio (evaluate your observed and pushforward densities)
        
        
    #     print('dimensions :  lambda = ' + str(lam.shape) + '   D = ' + str(D.shape) )
        # Perform KDE to estimate the pushforward
        # pf_dens = gauss_kde(D) # compute KDE estimate of it
        pf_dist = P.pushforward_dist
        
        # Specify Observed Measure - Uniform Density
        #obs_dist = sstats.uniform(0,uncertainty) # 1D only
        obs_dist = P.observed_dist
                
        # Solve the problem
        # r = obs_dists.pdf(D) / pf_dens.pdf(D) # vector of ratios evaluated at all the O(lambda)'s
        
        r = P.ratio
        M = np.max(r)
        eta_r = r/M
        
        if compare or smooth_post:
            if seed == 0:
                logging.info("""Performing Accept/Reject 
                to estimate the pushforward of the posterior.""")
            solve.problem(P, seed=seed)
            accept_inds = P.accept_inds
            num_accept = len(accept_inds)
            num_accept_list.append(num_accept)
            if num_accept < 10:
                logging.warn(("Less than ten samples were accepted for"
                    "`trial_seed` = %d Please increase the number of total"
                    "samples or the standard deviation.")%seed)
                smooth_flag = False
            else:
                smooth_flag = True
#         entropy_list.append( sstats.entropy( obs_dist.pdf(D), pf_dens.pdf(D) ) )    
        
        res = 50;
        max_x = D.max();
        # Plot stuff
        # plt.figure(1)
        x1 = np.linspace(-0.25, max_x, res)
        ax1.plot(x1, pf_dens.pdf(x1))
        plt.title('Pushforward Q(Prior)')
        plt.xlabel('Q(lambda)')
        
        x2 = np.linspace(0, max_x, res)
        ax2.plot(x2, obs_dist.pdf(x2))
        if compare:
            push_post_dens_kde = gauss_kde(D[:,accept_inds])
            pf = push_post_dens_kde.pdf(x2)
            ax2.plot(x2, pf, alpha=0.2)
#             ax2.legend(['Observed','Recovered'])
        plt.title('Observed Density')
        plt.xlabel('Q(lambda)')
        

        x3 = np.linspace(a,b, res)
        if smooth_post:
            input_dim = lam.shape[1] # get input dimension by observing the shape of lambda
            if smooth_flag:
                post_dens_kde = gauss_kde(lam[accept_inds,:].transpose())
                ps = post_dens_kde.pdf(x3)
                ax3.plot(x3, ps)
            
        else:    
            ax3.scatter(lam, eta_r)
        # plt.plot(lam_accept, gauss_kde(lam_accept))
        plt.scatter(lam_0, 0.05)
        plt.title('Posterior Distribution') #\nof Uniform Observed Density \nwith bound = %1.2e'%uncertainty)
        plt.xlabel('Lambda')
    #     plt.title('$\eta_r$')
        # # OPTIONAL:
        # pr = 0.2 # percentage view-window around true parameter.
    #     plt.xlim(lam0*np.array([1-pr,1+pr]))
        plt.xlim([a,b])
    
    plt.show()
    
#     print('\tMean Entropy is: %.2f with var %.2f'%(np.mean(entropy_list), np.std(entropy_list)))
#     print('Entropies: ')
    if compare or smooth_post:
#         print(['%.2f '%entropy_list[n] for n in range(num_trials)])
        print('Median Acceptance Rate: %2.2f%%'%(100*np.mean(num_accept_list)/num_samples) )
    #     return eta_r






def make_tabulated_sandbox(num_experiments=1):
    # We create many copies of the same widget objects in order to isolate our experimental areas.
    num_samples = [widgets.IntSlider(value=1500, continuous_update=False, 
        orientation='vertical', disable=False,
        min=int(5E2), max=int(2.5E4), step=500, 
        description='Samples $N$') for k in range(num_experiments)]

    sd = [widgets.FloatSlider(value=0.25, continuous_update=False, 
        orientation='vertical', disable=False,
        min=0.05, max=1.75, step=0.05, 
        description='$\sigma$') for k in range(num_experiments)]

    lam_min, lam_max = 2.0, 7.0
    lam_bound = [widgets.FloatRangeSlider(value=[3.0, 6.0], continuous_update=False, 
        orientation='horizontal', disable=False,
        min=lam_min, max = lam_max, step=0.5, 
        description='$\Lambda \in$') for k in range(num_experiments)]

    lam_0 = [widgets.FloatSlider(value=4.5, continuous_update=False, 
        orientation='horizontal', disable=False,
        min=lam_bound[k].value[0], max=lam_bound[k].value[1], step=0.1, 
        description='IC: $\lambda_0$') for k in range(num_experiments)]


    t_0 = [widgets.FloatSlider(value=0.5, continuous_update=False, 
        orientation='horizontal', disable=False,
        min=0.1, max=2.0, step=0.05,
        description='$t_0$ =', readout_format='.2f') for k in range(num_experiments)]

    Delta_t = [widgets.FloatSlider(value=0.1, continuous_update=False, 
        orientation='horizontal', disable=False,
        min=0.05, max=0.5, step=0.05,
        description='$\Delta_t$ =', readout_format='1.2e') for k in range(num_experiments)]

    num_observations = [widgets.IntSlider(value=50, continuous_update=False, 
        orientation='horizontal', disable=False,
        min=1, max=100, 
        description='# Obs. =') for k in range(num_experiments)]
    
    compare = [widgets.Checkbox(value=False, disable=False,
        description='Observed v. Q(Post)') for k in range(num_experiments)]
    
    smooth_post = [widgets.Checkbox(value=False, disable=False,
        description='Smooth Posterior') for k in range(num_experiments)]
    
    fixed_noise = [widgets.Checkbox(value=False, disable=False,
        description='Fixed Noise Model') for k in range(num_experiments)]
    
    num_trials = [widgets.IntSlider(value=1, continuous_update=False, 
        orientation='vertical', disable=False,
        min=1, max=25, 
        description='Num. Trials') for k in range(num_experiments)]
    
    # IF YOU ADD MORE FUNCTIONS to cb_sandbox.py, increase max below.
    fun_choice = [widgets.IntSlider(value=0, continuous_update=False, 
        orientation='horizontal', disable=False,
        min=0, max=1, 
        description='Fun. Choice') for k in range(num_experiments)]
    
    fixed_obs_window = [widgets.Checkbox(value=False, disable=False,
        description='Fixed Obs. Window') for k in range(num_experiments)]
                                
    Keys = [{'num_samples': num_samples[k], 
            'lam_bound': lam_bound[k], 
            'lam_0': lam_0[k], 
            't_0': t_0[k], 
            'Delta_t': Delta_t[k],
            'num_observations': num_observations[k], 
            'sd': sd[k],
            'compare': compare[k],
            'smooth_post': smooth_post[k],
            'fixed_noise': fixed_noise[k],
            'num_trials': num_trials[k], 
            'fun_choice': fun_choice[k]} for k in range(num_experiments)] 

    # Make all the interactive outputs for each tab and store them in a vector called out. (for output)
    out = [widgets.interactive_output(sandbox, Keys[k]) for k in range(num_experiments)]
    
    
    ### LINK WIDGETS TOGETHER (dependent variables) ###
    # if you change the bounds on the parameter space, update the bounds of lambda_0                          
    def update_lam_0(*args):
        k = tab_nest.selected_index
    #     lam_0[k].value = np.minimum(lam_0[k].value, lam_bound[k].value[1] )
    #     lam_0[k].value = np.maximum(lam_0[k].value, lam_bound[k].value[0] )
        lam_0[k].min = lam_bound[k].value[0] 
        lam_0[k].max = lam_bound[k].value[1]

    [lam_bound[k].observe(update_lam_0, 'value') for k in range(num_experiments)]
    
    
    current_window_size = [ num_observations[k].value*Delta_t[k].value for k in range(num_experiments)]
    def lock_window_size(*args): # if you want to lock the window
        k = tab_nest.selected_index
        if fixed_obs_window[k].value:
            current_window_size[k] = num_observations[k].value*Delta_t[k].value # record the present value for later use.
        
    def update_num_obs(*args): # update num obs if Delta_t changes
        k = tab_nest.selected_index
        if fixed_obs_window[k].value:
            num_observations[k].value = current_window_size[k]/Delta_t[k].value
    
    def update_delta_t(*args): # update num obs if Delta_t changes
        k = tab_nest.selected_index
        if fixed_obs_window[k].value:
            Delta_t[k].value = current_window_size[k]/num_observations[k].value
    
    [fixed_obs_window[k].observe(lock_window_size, 'value') for k in range(num_experiments)]
    [Delta_t[k].observe(update_num_obs, 'value') for k in range(num_experiments)]
    [num_observations[k].observe(update_delta_t, 'value') for k in range(num_experiments)]
    
    
    ### GENERATE USER INTERFACE ###
    lbl = widgets.Label("UQ Sandbox", disabled=False)
    # horizontal and vertical sliders are grouped together, displayed in one horizontal box.
    # This HBox lives in a collapsable accordion below which the results are displayed.
    h_sliders = [widgets.VBox([lam_bound[k], lam_0[k], 
                               t_0[k], Delta_t[k], 
                               num_observations[k] ]) for k in range(num_experiments) ]
    v_sliders = [widgets.HBox([ num_samples[k], num_trials[k],
                               sd[k] ]) for k in range(num_experiments) ]
    options = [ widgets.VBox([widgets.Text('Model Options', disabled=True), 
                              fixed_noise[k], fixed_obs_window[k], fun_choice[k],
                              widgets.Text('Plotting Options', disabled=True), 
                              compare[k], smooth_post[k]]) for k in range(num_experiments)]
    user_interface = [widgets.HBox([h_sliders[k], options[k], v_sliders[k]]) for k in range(num_experiments)]
    
    # format the widgets layout (non-default options)
    for k in range(num_experiments): 
        h_sliders[k].layout.justify_content = 'center'
        v_sliders[k].layout.justify_content = 'center'
        user_interface[k].layout.justify_content = 'center'

        
    ### MAKE TABULATED NOTEBOOK ###
    # Create our pages
    pages = [widgets.HBox() for k in range(num_experiments)]

    # instantiate notebook with tabs (accordions) representing experiments
    tab_nest = widgets.Tab()
    tab_nest.children = [pages[k] for k in range(num_experiments)]

    # title your notebooks
    experiment_names = ['Experiment %d'%k for k in range(num_experiments)]
    for k in range(num_experiments):
        tab_nest.set_title(k, experiment_names[k])

    # Spawn the children!!!
    for k in range(num_experiments):
    #     content = widgets.VBox( [user_interface[k], out[k]] )
        A = widgets.Accordion(children=[ user_interface[k] ])
        A.set_title(0,lbl.value)
        A.layout.justify_content = 'center'
        content = widgets.VBox([ A, out[k]  ])
        content.layout.justify_content = 'center'
        tab_nest.children[k].children = [content]
    
    return tab_nest, Keys, fixed_obs_window


def set_prop(num_experiments, K, prop, val):
    for k in range(num_experiments): # for each tab
        K[k][prop].value = val
