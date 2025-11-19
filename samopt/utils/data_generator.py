# data_generator.py

def CTMC(purse):#
    import numpy as np
    from copy import copy

    # Set up length of simulation and reporting times.
    T      = purse.T # reporting times
    tmin   = purse.tmin
    tmax   = purse.tmax
    maxiter = purse.maxiter
    mod_pars = purse.mod_pars
    init_cond = purse.init_cond
    trans_matrix = purse.trans_matrix
    # Set state variables equal to initial condition vector
    x  = [purse.init_cond] # This will be used to save the state values at the reporting times
    xt = np.array(init_cond) # This will be used to update the states at each event time
    
    # get first observation reporting time beyond tmin.
    j  = 1
    Treport = T[j]
    
    # start time
    t  = tmin
    ii = 0
    while (t < tmax) and (ii < maxiter):
        ii += 1    
        
        # Calculate the transition rates beta*S*I, alpha*I and store in an array.
        st = xt[0]
        it = xt[1]
        alpha = mod_pars[0]
        beta = mod_pars[1]
        rates = np.array([beta*st*it,alpha*it])

        # Sum the rates, and if it is positive, compute the cdf of the transition probabilities,
        # Choose the time of the next event and which event occurs,
        # Update the state vector and record the states at the reporting times.
        # Otherwise, set x to previous x.
        sum_rates = sum(rates)
        if sum_rates > 0:
            
            # Compute the cdf of the transition probabilities
            cdf = np.cumsum(rates/sum_rates)
        
            #Choose the time of next event
            k   = np.random.uniform(0,1) #Choose a uniform random number
            t  += np.random.exponential(1.0/sum_rates)
        
            # Choose event by finding where k falls within the array cdf.
            which_transition = np.where(k<=cdf)[0][0]
    
            # Add corresponding row of transition matrix to previous state array x,
            # but save previous states in x_prev.
            xt_prev = copy(xt)
            xt += trans_matrix[which_transition]
        
            # if time exceeds next observation reporting time, 
            # return data of previous state
            if t > Treport:
                x.append(list(xt_prev))
            
                # get next reporting time
                j += 1
                if j < len(T):
                    Treport = T[j]
                else:
                    print('I am breaking at A')
                    break
            
            # if time exceeds last reporting time, we're done!
            if t > T[-1]:
                print('I am breaking at B')
                break
                
                
        else:
            print('I am breaking at C')
            break 
            
    return np.array(x), xt