import numpy as np
import tensorly as tl
from copy import deepcopy

class tensor_PLSA:
    """
    Tensor Probablistic Latent Semantic Analysis
    
    A tensor extension of PLSA [1]
    
    It can also be understood as non-negative PARAFAC [2] with KL-divergence as loss
    
    Parameters
    ----------
    n_components : int, required
        number of components. It is also the number of columns in factor matrices.
        
    Attributes
    ----------
    nc : int,
        number of components
    
    params_set : boolen,
        whether attributes p_z and factors contain valid value
    
    shape : array-like, int type elements
        shape of the probability tensor
        exists after the parameter is set
        the first element of shape should be the number of possible outcomes
        for the first "axis" of tensor. In the case of a matrix, the first
        element of shape should be the number of rows.
    
    dim : int
        the number of axis of tensor, which is also len(shape)
        equal to 2 for a matrix
    
    p_z : numpy array of shape (nc,)
        prior probability of each component
    
    factors : list of numpy arrays, 
        length of list equal to len(shape)
        shape of the i-th array being (shape[i], nc)
        factors[a][i,j] = prob(X_{a}=i|Z=j)
        conditional probability of the a-th random variable being
        i given component being j
        factor matrices in terms of non-negative parafac
        
    References
    ----------
    ..[1] Hoffman, T. (1999). Probabilistic latent semantic indexing. 
    Proceedings of ACM SIGIR, 1999.
    
    ..[2] Shashua, A., & Hazan, T. (2005, August). Non-negative tensor
    factorization with applications to statistics and computer vision. 
    In Proceedings of the 22nd international conference on Machine 
    learning (pp. 792-799).
    """
    def __init__(self, n_components, eps = 1e-32):
        # TODO : check n_components is a positive integer
        self.nc = n_components
        self.params_set = False
        self.eps = eps
        
    def set_params(self, p_z, factors, tied_axis_indices = None, copy = True):
        """
        Set model parameters (attribute p_z and factors) to specific values
        
        Parameters
        ----------
        p_z : numpy array of shape (nc,) or array-like
            value to be assigned to attribute p_z
            see attribute p_z
        
        factors : list of numpy arrays
            value to be assigned to attribute factors
            see attribute factors
            
        tied_axis_indices : list, elements being list of non-negative
        integers
        each sub-list is a list of indices of axes whose
        parameters are tied (i.e. factor matrices are identical)
        for example, for a matrix with two axes (row and column)
        and we set tied_axis_indices = [0,1], we are seeking a 
        symmetric decomposition of that matrix
        
        copy : boolen, default = True
            if value is to be copied
            if not copied, later on changing the values to be assigned could
            lead to undesired change of model parameters
            
        Returns
        ------
        Nothing
        """
        # TODO : check p_z, factors, for shape, non-negativity and normalization,
        if type(p_z) == np.ndarray:
            if copy:
                self.p_z = p_z.copy()
            else:
                self.p_z = p_z
        else:
            self.p_z = np.array(p_z)
        if tied_axis_indices is None:
            if copy:
                self.factors = deepcopy(factors)
            else:
                self.factors = factors
        else:
            self.a2factor = get_map_axis_to_parameters(tied_axis_indices, len(factors))
            self.factors = []
            for axis in range(len(factors)):
                if self.a2factor[axis] == axis:
                    if copy:
                        self.factors.append(factors[axis].copy())
                    else:
                        self.factors.append(factors[axis])
                else:
                    self.factors.append(self.factors[self.a2factor[axis]])
        self.params_set = True
        self.shape = [factor.shape[0] for factor in self.factors]
        self.dim = len(self.shape)
        
    def set_random_params(self, shape, sigma = 1, tied_axis_indices = None, seed = None, \
        dist = "dirichlet"):
        """
        Set model parameters (attribute p_z and factors) to random values
        
        Parameters
        ----------
        shape : array-like with int type elements
            the shape of target tensor for which to generate random parameters
            after the operation, attribute shape should become shape
            
        sigma : float with positive value
            parameter influencing how dispersed the probabilities are
            when sigma = 0, all probabilities are identical
            when sigma is large, the probabilities tend to be concentrated
        
        tied_axis_indices : list, elements being list of non-negative
            integers
            each sub-list is a list of indices of axes whose
            parameters are tied (i.e. factor matrices are identical)
            for example, for a matrix with two axes (row and column)
            and we set tied_axis_indices = [0,1], we are seeking a 
            symmetric decomposition of that matrix
            
        seed : int
            random seed for generation

        dist : {"log_normal", "dirichlet"}
            distribution to draw random model parameters from
            log-normal :
                draw from log normal distribution and normalize. i.e.
                draw from random i.i.d gaussian distribution, with mean
                eqaul to 0 and standard deviation set to sigma, then
                take exponential of the gaussian random variables. After
                that the exponentials are normalized according as
                probabilities
            dirichlet :
                draw from dirichlet distribution
                for now the alphas are all set to 1, this corresponds to
                a flat distribution over the probability simplex
                it may make sense to add alpha as an optional argument
                in the future
        
        Returns
        -------
        Nothing
        """
        # TODO
        # check shape being a array-like of positive integers
        # catch seed type error when seed cannot be converted to int (TypeError)
        if seed is not None:
            np.random.seed(seed)
        self.factors = []
        if tied_axis_indices is None:
            for Ni in shape:
                if dist == "log-normal":
                    factor = np.exp(np.random.normal(size = (Ni,self.nc))*sigma)
                    factor /= factor.sum(axis=0, keepdims=True)
                else:
                    factor = np.random.dirichlet(np.ones(Ni),size=self.nc).T
                self.factors.append(factor)
        else:
            self.a2factor = get_map_axis_to_parameters(tied_axis_indices, len(shape))
            for axis in range(len(shape)):
                if self.a2factor[axis] == axis:
                    if dist == "log-normal":
                        factor = np.exp(np.random.normal(size = (shape[axis],self.nc))*sigma)
                        factor /= factor.sum(axis=0, keepdims=True)
                        #self.factors.append(factor)
                    else:
                        factor = np.random.dirichlet(np.ones(shape[axis]),size=self.nc).T
                    self.factors.append(factor)
                else:
                    self.factors.append(self.factors[self.a2factor[axis]])
        if dist == "log-normal":
            self.p_z = np.exp(np.random.normal(size=self.nc)*sigma)
            self.p_z /= self.p_z.sum()
        else:
            self.p_z = np.random.dirichlet(np.ones(self.nc))
        self.shape = [factor.shape[0] for factor in self.factors]
        self.dim = len(shape)
        self.params_set = True

    def initialize_parameter(self, init, shape, sigma, tied_axis_indices, seed, dist, verbose):
        """
        Initialize model parameters (attribute p_z and factors)
        
        Parameters
        ----------
        init : {"random", "existing"}
            the method to initialize parameters
            random :
                use set_random_params method
            existing :
                use existing value
        
        shape : array-like with int type elements
            the shape of target tensor for analysis
            attribute shape should be the same as shape
            if existing values of parameters are to be used
            
        sigma : float with positive value
            parameter influencing how dispersed the probabilities are for
            random generation
            when sigma = 0, all probabilities are identical
            when sigma is large, the probabilities tend to be concentrated
            ignored when init is not "random"
            
        tied_axis_indices : list, elements being list of non-negative
            integers
            each sub-list is a list of indices of axes whose
            parameters are tied (i.e. factor matrices are identical)
            for example, for a matrix with two axes (row and column)
            and we set tied_axis_indices = [0,1], we are seeking a 
            symmetric decomposition of that matrix
            
        seed : int
            random seed for random generation
            ignored when init is not "random"

        dist : {"log_normal", "dirichlet"}
            distribution to draw random model parameters from
            log-normal :
                draw from log normal distribution and normalize. i.e.
                draw from random i.i.d gaussian distribution, with mean
                eqaul to 0 and standard deviation set to sigma, then
                take exponential of the gaussian random variables. After
                that the exponentials are normalized according as
                probabilities
            dirichlet :
                draw from dirichlet distribution
                for now the alphas are all set to 1, this corresponds to
                a flat distribution over the probability simplex
                it may make sense to add alpha as an optional argument
                in the future
            
        verbose : boolen
            whether to enable screen output
            
        Returns
        -------
        Nothing
        """
        #TODO: add one option of initialize with value passed to it
        if verbose:
            print("Initializing parameters")
        if init == "random":
            self.set_random_params(shape, sigma, tied_axis_indices, seed, dist)
        if init == 'existing':
            if self.params_set == False:
                raise ValueError("Parameters not set")
            elif len(self.shape) != len(shape) or np.allclose(self.shape, shape) == False:
                raise ValueError("Tensor shape doesn't match parameter shape")
    
    def optimize_stop_by_train_likelihood(self, T, max_iter, tol, verbose):
        """
        Optimize model parameters (attribute p_z and factors) with EM iterations
        
        The stopping criterion is when the improve in log likelihood is 
        smaller than tol
        
        Parameters
        ----------
        T : numpy array of shape equal to attribute shape
            Tensor for optimization, each element should be count (or probability)
            of the corresponding outcome
            
        max_iter : int, value should be positive
            The maximum number of iterations to run.
            If reached, the optimization is stopped even if the stopping criterion
            if not met
        
        tol : float, value should be non-negative
            tolerance for log likelihood increment
            If the increment is smaller than tolerance, optimization is stopped
            
        verbose : bool
            whether to enable screen output
            if so, training likelihood at each iteration would be displayed
            in addition to how the optimization stopped 
            (from reaching stopping criterion or exceeding limit on iteration)
            
        Returns
        -------
        Ls : list 
            a list of log likelihood at each iteration
            the first element is the log likelihood from initial parameter without
            fitting
            
        """
        
        # calculate initial log likelihood
        
        #Lnew would be used for the new log likelihood
        #i.e. after the values are properly updated, for each iteration
        Lnew = np.sum(T*np.log(tl.cp_to_tensor([self.p_z,self.factors])+self.eps))
        if verbose:
            print("initial training likelihood: ", np.round(Lnew,2))
        
        # optimizing
        
        #Ls the list of log likelihood for each iteration
        #including the initial one before optimization
        Ls = [Lnew,]
        #PzGelm is the probability of latent state z given which element it is
        #i.e. given a element at specific position in tensor, the probability
        #that it is from component z
        #stored in numpy array with shape (nc,shape[0],shape[1],..)
        PzGelm = np.zeros([self.nc,]+list(T.shape))
        
        #i would be iteraion index
        for i in range(max_iter):
            
            # E step
            
            #fi would be factor index
            #which is also the index of component (or z value)
            for fi in range(self.nc):
                #before normalization, it is joint probability 
                PzGelm[fi] = tl.cp_to_tensor([self.p_z[[fi]],\
                             [factor[:,fi].reshape(-1,1) for factor in self.factors]])
            #after normalization, it is the conditional probability
            PzGelm /= PzGelm.sum(axis=0) + self.eps
            #TPzGelm = tensor * Prob(z|element)
            TPzGelm = T*PzGelm
            #axis is the axis/view of tensor
            #for a matrix, axis 0 and 1 would be row and column
            
            # M step
            
            #update factor matrices
            #note factors are defined as probability of attribute given component z
            #axis is the axis/dimension/view of tensor
            #for a matrix axis=0 and 1 correspond to row and column
            for axis in range(self.dim):
                self.factors[axis][:] = 0
            
            for axis in range(self.dim):
                self.factors[axis] += TPzGelm.sum(axis=tuple(a for a in range(1,self.dim+1) \
                                                             if a != axis+1)).T
                
            for axis in range(self.dim):
                self.factors[axis] /= self.factors[axis].sum(axis=0, keepdims=True) + self.eps
            #update prior probability of z
            self.p_z = TPzGelm.sum(axis=tuple(range(1,self.dim+1)))
            self.p_z /= self.p_z.sum() + self.eps
            
            # evaluate training log likelihood
            
            Lold = Lnew
            Lnew = np.sum(T*np.log(tl.cp_to_tensor([self.p_z,self.factors])+self.eps))
            if verbose:
                print("iteration "+str(i)+" training likelihood: "+str(np.round(Lnew,2)))
            Ls.append(Lnew)
            
            # check stopping criterion
            
            #improve of log likelihood from last iteration
            diff = Lnew - Lold
            #if improve is too small, stop early
            if diff < tol:
                break
        if verbose:
            if diff < tol:
                print("Early stop at iteration " + str(i))
            else:
                print("Exceeding max_iter. Convergence not reached yet.")
        self.iter = i
        return Ls
    
    def optimize_stop_by_valid_likelihood(self, T, max_iter, T_valid, N_decrs, verbose):
        """
        Optimize model parameters (attribute p_z and factors) with EM iterations
        
        The stopping criterion is when the improve in log likelihood is 
        smaller than tol
        
        Parameters
        ----------
        T : numpy array of shape equal to attribute shape
            Tensor for optimization, each element should be count (or probability)
            of the corresponding outcome
            
        max_iter : int, value should be positive
            The maximum number of iterations to run.
            If reached, the optimization is stopped even if the stopping criterion
            if not met
            
        T_valid : numpy array of shape equal to attribute shape
            Tensor for validation, each element should be count (or probability)
            of the corresponding outcome
        
        N_decrs : int, value should be positive
            the maximum number of consecutive decrease in validation set log
            likelihood
            if exceeded, the optimization would stop
            
        verbose : bool
            whether to enable screen output
            if so, valid likelihood at each iteration would be displayed
            in addition to how the optimization stopped 
            (from reaching stopping criterion or exceeding limit on iteration)
            
        Returns
        -------
        Ls : list 
            a list of log likelihood at each iteration
            the first element is the log likelihood from initial parameter without
            fitting
            
        """
        # calculate initial log likelihood
        
        #Lnew would be used for the new log likelihood
        #i.e. after the values are properly updated, for each iteration
        Lnew = np.sum(T_valid*np.log(tl.cp_to_tensor([self.p_z,self.factors])+self.eps))
        if verbose:
            print("initial validation likelihood: ", np.round(Lnew,2))
            
        
        
        # optimizing
        
        #Ls the list of log likelihood for each iteration
        #including the initial one before optimization
        Ls = [Lnew,]
        #PzGelm is the probability of latent state z given which element it is
        #i.e. given a element at specific position in tensor, the probability
        #that it is from component z
        #stored in numpy array with shape (nc,shape[0],shape[1],..)
        PzGelm = np.zeros([self.nc,]+list(T.shape))
        #n_decrs is the number of consecutive decrease in valid dataset log
        #likelihood
        #it is set 0 in the beginning, reset to 0 when there is an incraese
        n_decrs = 0
        
        #i would be iteraion index
        for i in range(max_iter):
            
            # E step
            
            #fi would be factor index
            #which is also the index of component (or z value)
            for fi in range(self.nc):
                #before normalization, it is joint probability 
                PzGelm[fi] = tl.cp_to_tensor([self.p_z[[fi]],\
                             [factor[:,fi].reshape(-1,1) for factor in self.factors]])
            #after normalization, it is the conditional probability
            PzGelm /= PzGelm.sum(axis=0) + self.eps
            #TPzGelm = tensor * Prob(z|element)
            TPzGelm = T*PzGelm
            #axis is the axis/view of tensor
            #for a matrix, axis 0 and 1 would be row and column
            
            # M step
            
            #update factor matrices
            #note factors are defined as probability of attribute given component z
            #axis is the axis/dimension/view of tensor
            #for a matrix axis=0 and 1 correspond to row and column
            for axis in range(self.dim):
                self.factors[axis][:] = 0
                
            for axis in range(self.dim):
                self.factors[axis] += TPzGelm.sum(axis=tuple(a for a in range(1,self.dim+1) \
                                                             if a != axis+1)).T
                
            for axis in range(self.dim):
                self.factors[axis] /= self.factors[axis].sum(axis=0, keepdims=True) + self.eps
            #update prior probability of z
            self.p_z = TPzGelm.sum(axis=tuple(range(1,self.dim+1)))
            self.p_z /= self.p_z.sum() + self.eps
            
            # evaluate training log likelihood
            
            Lold = Lnew
            Lnew = np.sum(T_valid*np.log(tl.cp_to_tensor([self.p_z,self.factors])+self.eps))
            if verbose:
                print("iteration "+str(i)+" validation likelihood: "+str(np.round(Lnew,2)))
            Ls.append(Lnew)
            
            # check stopping criterion
            
            if Lnew < Lold:#if decrease
                n_decrs += 1#count one more decrease
                if n_decrs > N_decrs:#if beyond threshold
                    break#stop optimization
            else:#if increase
                n_decrs = 0#reset
        self.iter = i
        if verbose:
            if n_decrs > N_decrs:
                print("Early stop at iteration " + str(i))
            else:
                print("Exceeding max_iter. Convergence not reached yet")
        return Ls
    
    def fit(self, T, max_iter = 300, init = "random", sigma = 1, tied_axis_indices = None, \
            seed = None, stop_criterion = "train", tol = 1e-6, T_valid = None, N_decrs=2, \
            dist = "dirichlet", verbose = True):
        """
        fit model using EM algorithm and tensor T
        
        Parameters
        ----------
        T : numpy array of shape equal to attribute shape
            Tensor for optimization, each element should be count (or probability)
            of the corresponding outcome
            
        max_iter : int, default = 300, value should be positive
            The maximum number of iterations to run.
            If reached, the optimization is stopped even if the stopping criterion
            if not met
            
        init : {"random", "existing"}, default = "random"
            the method to initialize parameters
            random :
                use set_random_params method
            existing :
                use existing value
        
        sigma : float with positive value, default = 1
            parameter influencing how dispersed the probabilities are for
            random generation
            when sigma = 0, all probabilities are identical
            when sigma is large, the probabilities tend to be concentrated
            ignored when init is not "random"
        
        tied_axis_indices : list, elements being list of non-negative
            integers
            each sub-list is a list of indices of axes whose
            parameters are tied (i.e. factor matrices are identical)
            for example, for a matrix with two axes (row and column)
            and we set tied_axis_indices = [0,1], we are seeking a 
            symmetric decomposition of that matrix
            
        seed : int
            random seed for random generation
            ignored when init is not "random"
            
        stop_criterion : {"train","valid"}
            stopping criterion for optimization
            train :
                stop when the increment of log likelihood on training data (i.e. T)
                is below certain threshold (specified by tol)
            test :
                stop when the log likelihood on validation data (i.e. T_valid)
                keep decreasing for iterations beyong certain threshold (i.e. N_decrs)
        
        tol : float, value should be non-negative
            tolerance for log likelihood increment
            If the increment is smaller than tolerance, optimization is stopped
            Only used if stop_criterion is "train"
        
        T_valid : numpy array of shape equal to attribute shape
            Tensor for validation, each element should be count (or probability)
            of the corresponding outcome
            Only used if stop_criterion is "valid"
        
        N_decrs : int, value should be positive
            the maximum number of consecutive decrease in validation set log
            likelihood
            if exceeded, the optimization would stop
            only used if stop_criterion is "valid"

        dist : {"log_normal", "dirichlet"}
            distribution to draw random model parameters from
            log-normal :
                draw from log normal distribution and normalize. i.e.
                draw from random i.i.d gaussian distribution, with mean
                eqaul to 0 and standard deviation set to sigma, then
                take exponential of the gaussian random variables. After
                that the exponentials are normalized according as
                probabilities
            dirichlet :
                draw from dirichlet distribution
                for now the alphas are all set to 1, this corresponds to
                a flat distribution over the probability simplex
                it may make sense to add alpha as an optional argument
                in the future
            
        verbose : bool
            whether to enable screen output
            if so, log likelihood at each iteration would be displayed
            in addition to how the optimization stopped 
            (from reaching stopping criterion or exceeding limit on iteration)
            

            
        Returns
        -------
        Ls : list 
            a list of log likelihood at each iteration
            the type of log likelihood (from train or valid, or from T or T_valid)
            depends on stop_criterion
            the first element is the log likelihood from initial parameter without
            fitting
        """
        self.initialize_parameter(init, T.shape, sigma, tied_axis_indices, seed, dist, verbose)
        if stop_criterion == "train":
            return self.optimize_stop_by_train_likelihood(T, max_iter, tol, verbose)
        elif stop_criterion == "valid":
            return self.optimize_stop_by_valid_likelihood(T, max_iter, T_valid, N_decrs, verbose)
        
    def get_log_likelihood(self, T):
        """
        calculating log likelihood on tensor T
        
        Parameters
        ----------
        T : numpy array of shape equal to attribute shape
            Tensor to calculate log likelihood, 
            each element should be the count (or probability) of the corresponding outcome
        
        Returns
        -------
        LL : float
            log likelihood calculated on tensor T
        """
        if self.params_set:
            LL = np.sum(T*np.log(tl.cp_to_tensor([self.p_z,self.factors])+self.eps))
            return LL
        else:
            raise ValueError("parameters not set")

    def sample(self, N):
        """
        sample count tensor from current model
        
        Parameters
        ----------
        N : int, value should be positive
            total number of counts
            
        Returns
        -------
        T_sample : numpy array of shape equal to attribute shape
            each element is the count (or probability) of the corresponding outcome
        """
        if self.params_set:
            p = tl.cp_to_tensor([self.p_z,self.factors]).reshape(-1)
            T_sample = np.random.multinomial(N, p).reshape(self.shape)
            return T_sample
        else:
            raise ValueError("parameters not set")

    def get_tensor(self):
        """
        get the probability tensor specified by the model

        Parameters
        ----------
        Nothing

        Returns
        -------
        Nothing
        """
        return tl.cp_to_tensor([self.p_z,self.factors])


def get_map_axis_to_parameters(tied_axis_indices, dim):
    """
    construct dictionary mapping axis index to the smallest
    index which shares the same parameter
    
    Parameters
    ----------
    tied_axis_indices : list, elements being list of non-negative
        integers
        each sub-list is a list of indices of axes whose
        parameters are tied (i.e. factor matrices are identical)
        for example, for a matrix with two axes (row and column)
        and we set tied_axis_indices = [0,1], we are seeking a 
        symmetric decomposition of that matrix
    
    dim : int, value should be positive
        the dimensionality of the tensor, or the number of axes
        
    Returns
    -------
    a2factor : dictionary
        map each axis index to the smallest index sharing the
        same parameters
    """
    a2factor = {}
    for i in range(dim):
        a2factor[i] = i
    for indices in tied_axis_indices:
        sorted_indices = sorted(indices)
        for index in sorted_indices[1:]:
            a2factor[index] = sorted_indices[0]
    return a2factor
