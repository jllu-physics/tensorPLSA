import numpy as np
from tensorly.tenalg import khatri_rao
from copy import deepcopy
import tensorly as tl
import multiprocessing as mp

def calculate_count_z_MP_target(pid, indices, counts, p_z, factors, block_size, count_z_dict):
    """
    Calculate expected count attributed to each component z
    
    Part of M-setp in EM algorithm
    
    Parameters
    ----------
    pid : int, non-negative
        process index for multiprocessing
        also the key when delivering result through dictionary
        
    indices : numpy array, of shape (N,dim) (see below)
        indices for the corresponding count
        we store a count tensor sparsely by agreeing that element
        at index indices[i] has value (or count) counts[i]
        In this case, N above would be the number of non-zero
        elements, dim would be the number of different axis for
        tensor (i.e., the number of integers required to locate
        an element; for a matrix, dim=2)
        
    counts : numpy array, of shape (N,) (see below)
        counts for the corresponding indices
        we store a count tensor sparsely by agreeing that element
        at index indices[i] has value (or count) counts[i]
        In this case, N above would be the number of non-zero
        elements.
        
    p_z : numpy array of shape (nc,)
        nc would be the number components in the corresponding
        tensor PLSI model
        prior probability of each component
    
    factors : list of numpy arrays, 
        length of list equal to dim
        shape of the i-th array being (max(indices[:,i]), nc)
        nc would be the number components in the corresponding
        tensor PLSI model
        factors[a][i,j] = prob(X_{a}=i|Z=j)
        conditional probability of the a-th random variable being
        i given component being j
        factor matrices in terms of non-negative parafac
        
    block_size : int, value should be positive
        the maximum number of elements to calculate each time
        in the extreme of 1, this means calculating element by
        element, could be slow
        in the other extreme of N (number of nonzero elements),
        this means calculating all at one, could be memory
        intensive
        
    count_z_dict : dictionary to store results
        the process assigned pid i should use the value stored
        with key i
        
    """
    
    # N, length of indices, also the number of non-zero elements
    # in the original count tensor
    N = len(indices)
    # nc, the number of components
    _,nc = factors[0].shape
    # dim, the number of axis of tensor, 2 for matrix
    dim = len(factors)
    # count_z is theexpected count attributable to each component
    # z, for each tensor element (i.e. count)
    count_z = np.zeros((N,nc))
    count_z[:] = p_z
    # n_block is the number of times we need to iterate before
    # taking care of all data
    n_block = int((N-1)//block_size) + 1
    
    #b for block index
    for b in range(n_block):
        #s is starting index
        s = b*block_size
        #e is ending index, not inclusive
        e = (b+1)*block_size
        #d for axis index (for matrix, 0 for row etc)
        for d in range(dim):
            count_z[s:e] *= factors[d][indices[s:e,d]]
        count_z[s:e] /= count_z[s:e].sum(axis=1, keepdims=True)
        count_z[s:e] *= counts[s:e].reshape(-1,1)
    #store count_z in corresponding value, key being pid
    count_z_dict[pid] = count_z

def calculate_probability_MP_target(pid, indices, counts, p_z, factors, block_size, probs_dict):
    """
    Calculate probability observing an outcome specified by indices
    
    Parameters
    ----------
    pid : int, non-negative
        process index for multiprocessing
        also the key when delivering result through dictionary
        
    indices : numpy array, of shape (N,dim) (see below)
        indices for the corresponding count
        we store a count tensor sparsely by agreeing that element
        at index indices[i] has value (or count) counts[i]
        In this case, N above would be the number of non-zero
        elements, dim would be the number of different axis for
        tensor (i.e., the number of integers required to locate
        an element; for a matrix, dim=2)
        
    counts : numpy array, of shape (N,) (see below)
        counts for the corresponding indices
        we store a count tensor sparsely by agreeing that element
        at index indices[i] has value (or count) counts[i]
        In this case, N above would be the number of non-zero
        elements.
        
    p_z : numpy array of shape (nc,)
        nc would be the number components in the corresponding
        tensor PLSI model
        prior probability of each component
    
    factors : list of numpy arrays, 
        length of list equal to dim
        shape of the i-th array being (max(indices[:,i]), nc)
        nc would be the number components in the corresponding
        tensor PLSI model
        factors[a][i,j] = prob(X_{a}=i|Z=j)
        conditional probability of the a-th random variable being
        i given component being j
        factor matrices in terms of non-negative parafac
        
    block_size : int, value should be positive
        the maximum number of elements to calculate each time
        in the extreme of 1, this means calculating element by
        element, could be slow
        in the other extreme of N (number of nonzero elements),
        this means calculating all at one, could be memory
        intensive
        
    probs_dict : dictionary to store results
        the process assigned pid i should use the value stored
        with key i
        
    """
    
    # N, length of indices, also the number of non-zero elements
    # in the original count tensor
    N = len(indices)
    # nc, the number of components
    _,nc = factors[0].shape
    # dim, the number of axis of tensor, 2 for matrix
    dim = len(factors)
    # prob_z is the probability observing an outcome specified by
    # the indices, and the latent component is z
    prob_z = np.zeros((block_size,nc))
    # probs has element of probability observing an outcome
    # specified by the indices (with any component z)
    probs = np.zeros(N)
    # n_block is the number of times we need to iterate before
    # taking care of all data
    n_block = int((N-1)//block_size) + 1
    
    #b for block index
    for b in range(n_block):
        prob_z[:] = p_z
        #s is starting index
        s = b*block_size
        #e is ending index, not inclusive
        e = min((b+1)*block_size, N)
        #size is the number of elements in the block, i.e
        #length of the block
        #usually it equals to block_size, but it is different
        #for the last block, which is often smaller
        size = e-s
        #d for axis index (for matrix, 0 for row etc)
        for d in range(dim):
            prob_z[:size] *= factors[d][indices[s:e,d]]
        #size is importnat here, without it, there could be
        #shape mismatch
        probs[s:e] = prob_z[:size].sum(axis=1)
        #count_z[s:e] /= count_z[s:e].sum(axis=1, keepdims=True)
        #count_z[s:e] *= counts[s:e].reshape(-1,1)
    #store count_z in corresponding value, key being pid
    probs_dict[pid] = probs

class Tensor_PLSA_Sparse:
    """
    Tensor Probablistic Latent Semantic Analysis with Sparse Count Tensor
    
    A tensor extension of PLSA [1]
    
    It can also be understood as non-negative PARAFAC [2] with KL-divergence as loss
    
    Parameters
    ----------
    n_components : int, required
        number of components. It is also the number of columns in factor matrices.
        
    block_size : int, value should be positive, default = 128
        maximum number of tensor elements to use for sparse parameter update each time.
        
    n_proc : int, value should be positive, default = 4
        number of processes used to accelerate the computation
        
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
    def __init__(self, n_components, block_size = 128, n_proc = 4):
        # TODO : check n_components is a positive integer
        self.nc = n_components
        self.params_set = False
        self._aux_array_prepared = False
        self.block_size = block_size
        self.n_proc = n_proc
        self._parameters_tied = False
        
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
            parameters are tied
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
            self._parameters_tied = True
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
        
    def set_random_params(self, shape, sigma = 1, tied_axis_indices = None, seed = None, dist = "dirichlet"):
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
            parameters are tied
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
                factor = np.exp(np.random.normal(size = (Ni,self.nc))*sigma)
                factor /= factor.sum(axis=0, keepdims=True)
                self.factors.append(factor)
        else:
            self._parameters_tied = True
            self.a2factor = get_map_axis_to_parameters(tied_axis_indices, len(shape))
            for axis in range(len(shape)):
                if self.a2factor[axis] == axis:
                    if dist == "log-normal":
                        factor = np.exp(np.random.normal(size = (shape[axis],self.nc))*sigma)
                        factor /= factor.sum(axis=0, keepdims=True)
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

    def initialize_parameter(self, init, shape, sigma=1, tied_axis_indices=None, seed=None, dist='dirichlet', verbose=False):
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
            parameters are tied
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
                
    def prepare_fitting(self, eps = 1e-16):
        """
        Prepare auxillary arrays and objects used in fitting
        
        Parameters
        ----------
        eps : float, value should be positive
            epsilon, the tiny positive real number to guard against
            numerical errors such as division by zero
            
        Returns
        -------
        Nothing
        """
        #if arrays are not prepared (otherwise, do nothing)
        if self._aux_array_prepared == False:
            #TODO: add check shape
            #check if the shapes are consistent when
            #aux_array_prepared is True
            #allocate memory for new_factors
            #a list of arrays to store new factor matrices
            self.new_factors = []
            if self._parameters_tied:
                for axis in range(self.dim):
                    if self.a2factor[axis] == axis:
                        self.new_factors.append(np.zeros((self.shape[axis], self.nc)))
                    else:
                        self.new_factors.append(self.new_factors[self.a2factor[axis]])
            else:
                for n in self.shape:
                    self.new_factors.append(np.zeros((n, self.nc)))
            #allocate for new_p_z
            #stores p_z update
            self.new_p_z = np.zeros(self.nc)
            #epsilon, guard against division by zero
            self.eps = eps
            #multiprocessing manager
            self.manager = mp.Manager()
            #dictionary to store results of count attributed to
            #each component z
            self.count_z_dict = self.manager.dict()
            #dictionary to store result of probability of each
            #observed outcome
            self.probs_dict = self.manager.dict()
            self._aux_array_prepared = True

    def calculate_count_attributable_to_z(self, indices, counts):
        """
        Calculate count of outcomes attributable to component z
        
        Parameters
        ----------
        indices : numpy array, of shape (N,dim) (see below)
            one of the two variables used to sparsely store tensor
            used to calculate count
            the other variable is counts
            indices for the corresponding count
            we store a count tensor sparsely by agreeing that element
            at index indices[i] has value (or count) counts[i]
            In this case, N above would be the number of non-zero
            elements, dim would be the number of different axis for
            tensor (i.e., the number of integers required to locate
            an element; for a matrix, dim=2)
        
        counts : numpy array, of shape (N,) (see below)
            one of the two variables used to sparsely store tensor
            used to calculate count
            the other variable is indices
            counts for the corresponding indices
            we store a count tensor sparsely by agreeing that element
            at index indices[i] has value (or count) counts[i]
            In this case, N above would be the number of non-zero
            elements.
            
        Returns
        -------
        count_z : numpy array of shape (L, nc)
            where L is the length of indices (or number of rows)
            nc is the number of components
            count arrtibutable to each component z
            each row corresponds to a specific outcome specified
            by indices
            each column corresponds to a component
            the elements should be non-negative, and the i-th row
            sum up to counts[i]
        """
        L = len(indices)#total observations passed to us
        batch_size = int(np.ceil(L/self.n_proc))#work load of each process
        #calculate with multiple processes
        prcs = []
        for i in range(self.n_proc):
            prcs.append(\
                mp.Process(target = calculate_count_z_MP_target, \
                    args=(i,indices[i*batch_size:(i+1)*batch_size], counts[i*batch_size:(i+1)*batch_size], \
                          self.p_z, self.factors, self.block_size, self.count_z_dict)))
            prcs[i].start()
        for i in range(self.n_proc):
            prcs[i].join()
        #put results from multiple processes together to get back the whole
        #result. It involves creating a new array and copy value to it
        #we can save resources by avoid copying and use the dictionary directly
        count_z = np.vstack([self.count_z_dict[i] for i in range(self.n_proc)])
        return count_z
    
    def fit(self, indices, counts, max_iter = 300, init = "random", sigma = 1, tied_axis_indices = None, \
            seed = None, stop_criterion = "train", tol = 1e-6, indices_valid = None, counts_valid = None, \
            N_decrs=4, eps = 1e-32, back_up = False, sparse_update = True, dist = "dirichlet", verbose = True):
        """
        fit model using EM algorithm and tensor stored sparsely in
        the form of indices and counts
        
        Parameters
        ----------
        indices : numpy array, of shape (N,dim) (see below)
            one of the two variables used to sparsely store tensor
            used to fit model
            the other variable is counts
            indices for the corresponding counts
            we store a count tensor sparsely by agreeing that element
            at index indices[i] has value (or count) counts[i]
            In this case, N above would be the number of non-zero
            elements, dim would be the number of different axis for
            tensor (i.e., the number of integers required to locate
            an element; for a matrix, dim=2)
        
        counts : numpy array, of shape (N,) (see below)
            one of the two variables used to sparsely store tensor
            used to fit model
            the other variable is indices
            counts for the corresponding indices
            we store a count tensor sparsely by agreeing that element
            at index indices[i] has value (or count) counts[i]
            In this case, N above would be the number of non-zero
            elements.
            
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
            parameters are tied
            for example, for a matrix with two axes (row and column)
            and we set tied_axis_indices = [0,1], we are seeking a 
            symmetric decomposition of that matrix
            
        seed : int, default is None
            random seed for random generation
            ignored when init is not "random"
            
        stop_criterion : {"train","valid"}, default is "train"
            stopping criterion for optimization
            train :
                stop when the increment of log likelihood on training data 
                (i.e. indices and counts)
                is below certain threshold (specified by tol)
            test :
                stop when the log likelihood on validation data 
                (i.e. indices_valid and counts_valid)
                keep decreasing for iterations beyong certain threshold (i.e. N_decrs)
        
        tol : float, value should be non-negative, default is 1e-6
            tolerance for log likelihood increment
            If the increment is smaller than tolerance, optimization is stopped
            Only used if stop_criterion is "train"
        
        indices_valid : numpy array, of shape (N,dim) (see below)
            default is None
            one of the two variables used to sparsely store tensor
            used to validate model
            the other being counts_valid
            indices for the corresponding count
            we store a count tensor sparsely by agreeing that element
            at index indices[i] has value (or count) counts[i]
            In this case, N above would be the number of non-zero
            elements, dim would be the number of different axis for
            tensor (i.e., the number of integers required to locate
            an element; for a matrix, dim=2)
        
        counts_valid : numpy array, of shape (N,) (see below)
            default is None
            one of the two variables used to sparsely store tensor
            used to validate model
            the other being indices_valid
            counts for the corresponding indices
            we store a count tensor sparsely by agreeing that element
            at index indices[i] has value (or count) counts[i]
            In this case, N above would be the number of non-zero
            elements.
        
        N_decrs : int, value should be positive, default is 4
            the maximum number of consecutive decrease in validation set log
            likelihood
            if exceeded, the optimization would stop
            only used if stop_criterion is "valid"
            the default value of 4 roughly correspond to .05 significant
            level that the log likelihood is decreasing
            The null hypothesis is that the optimization has plateaued,
            and any decrease is due to fluctuation, then the chance of
            decrease is .5. 
            Under null the chance we observe more than 4 consecutive
            decrease is 0.5^5<0.05 (from geometric dist)
            
        back_up : bool, default = False
            whether to store the best set of parameters we have seen
            at each iteration
            if True,  the set of parameters with best validation set
            log likelihood would be returned, but additional resource
            would be required to store the best parameters
            if False, the last set of parameters would be returned,
            which has sub-optimal performance on validation set, but
            the optimization would run faster
            
        sparse_update : bool, default = True
            whether to use sparse update for adding counts to the
            factor matrices
            if True, update element by element. This is recommended
            when the factor matrices are expected to be sparse, as
            it would update less elements
            if False, update by constructing a new matrix for each
            factor matrix. This is recommended if the factor matrices
            are expected to be dense, as it can vectorization to
            speed up the ccomputation
        
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
            the type of log likelihood (from train or valid) depends on 
            stop_criterion
            the first element is the log likelihood from initial parameter without
            fitting
        """
        shape = np.max(indices, axis=0).astype(int) + 1
        self.initialize_parameter(init, shape, sigma, tied_axis_indices, seed, dist, verbose)
        self.prepare_fitting(eps)
        if stop_criterion == "train":
            return self.optimize_stop_by_train_likelihood(indices, counts, max_iter, tol, \
                                                          sparse_update, verbose)
        elif stop_criterion == "valid":
            return self.optimize_stop_by_valid_likelihood(indices, counts, max_iter, \
                indices_valid, counts_valid, N_decrs, back_up, sparse_update, verbose)
    
    def get_log_likelihood(self, indices, counts):
        """
        Calculate count of outcomes attributable to component z
        
        Parameters
        ----------
        indices : numpy array, of shape (N,dim) (see below)
            one of the two variables used to sparsely store tensor
            used to calculate count
            the other variable is counts
            indices for the corresponding count
            we store a count tensor sparsely by agreeing that element
            at index indices[i] has value (or count) counts[i]
            In this case, N above would be the number of non-zero
            elements, dim would be the number of different axis for
            tensor (i.e., the number of integers required to locate
            an element; for a matrix, dim=2)
        
        counts : numpy array, of shape (N,) (see below)
            one of the two variables used to sparsely store tensor
            used to calculate count
            the other variable is indices
            counts for the corresponding indices
            we store a count tensor sparsely by agreeing that element
            at index indices[i] has value (or count) counts[i]
            In this case, N above would be the number of non-zero
            elements.
            
        Returns
        -------
        count_z : numpy array of shape (L, nc)
            where L is the length of indices (or number of rows)
            nc is the number of components
            count arrtibutable to each component z
            each row corresponds to a specific outcome specified
            by indices
            each column corresponds to a component
            the elements should be non-negative, and the i-th row
            sum up to counts[i]
        """
        L = len(indices)#total observations passed to us
        batch_size = int(np.ceil(L/self.n_proc))#work load of each process
        #calculate with multiple processes
        prcs = []
        for i in range(self.n_proc):
            prcs.append(mp.Process(target = calculate_probability_MP_target, \
                args=(i,indices[i*batch_size:(i+1)*batch_size], counts[i*batch_size:(i+1)*batch_size], \
                    self.p_z, self.factors, self.block_size, self.probs_dict)))
            prcs[i].start()
        for i in range(self.n_proc):
            prcs[i].join()
        #put together the probabilities calculated from each process
        #this involves allocating new memory and copying
        #could be improved if the two lines below are merged
        #i.e. conctenate(log(prob_dicts+eps))
        probs = np.concatenate([self.probs_dict[i] for i in range(self.n_proc)])
        #print(len(probs))
        LL = counts.dot(np.log(probs+self.eps))
        return LL

    def optimize_once(self, indices, counts, sparse_update):
        self.accumulate(indices, counts, sparse_update)
        self.normalize()
        self.update_value()
        
    def optimize_stop_by_train_likelihood(self, indices, counts, max_iter, tol, sparse_update, verbose):
        """
        optimize model parameter using EM algorithm and tensor
        stored sparsely in the form of indices and counts
        
        optimization stops when the increment in log likelihood
        on training set (i.e. tensor for fitting) is below a
        certain tolerance
        
        Parameters
        ----------
        indices : numpy array, of shape (N,dim) (see below)
            one of the two variables used to sparsely store tensor
            used to fit model
            the other variable is counts
            indices for the corresponding counts
            we store a count tensor sparsely by agreeing that element
            at index indices[i] has value (or count) counts[i]
            In this case, N above would be the number of non-zero
            elements, dim would be the number of different axis for
            tensor (i.e., the number of integers required to locate
            an element; for a matrix, dim=2)
        
        counts : numpy array, of shape (N,) (see below)
            one of the two variables used to sparsely store tensor
            used to fit model
            the other variable is indices
            counts for the corresponding indices
            we store a count tensor sparsely by agreeing that element
            at index indices[i] has value (or count) counts[i]
            In this case, N above would be the number of non-zero
            elements.
            
        max_iter : int, default = 300, value should be positive
            The maximum number of iterations to run.
            If reached, the optimization is stopped even if the stopping criterion
            if not met
        
        tol : float, value should be non-negative
            tolerance for log likelihood increment
            If the increment is smaller than tolerance, optimization is stopped
            Only used if stop_criterion is "train"
            
        sparse_update : bool
            whether to use sparse update for adding counts to the
            factor matrices
            if True, update element by element. This is recommended
            when the factor matrices are expected to be sparse, as
            it would update less elements
            if False, update by constructing a new matrix for each
            factor matrix. This is recommended if the factor matrices
            are expected to be dense, as it can vectorization to
            speed up the ccomputation
            
        verbose : bool
            whether to enable screen output
            if so, log likelihood at each iteration would be displayed
            in addition to how the optimization stopped 
            (from reaching stopping criterion or exceeding limit on iteration)
            
        Returns
        -------
        Ls : list 
            a list of log likelihood at each iteration
            the type of log likelihood (from train or valid) depends on 
            stop_criterion
            the first element is the log likelihood from initial parameter without
            fitting
        """
        #N = len(indices)
        #n_block = int( (N-1) // self.block_size ) + 1
        # calculate initial log likelihood
        
        #Lnew would be used for the new log likelihood
        #i.e. after the values are properly updated, for each iteration
        Lnew = self.get_log_likelihood(indices, counts)
        if verbose:
            print("initial training likelihood: ", np.round(Lnew,2))
        
        # optimizing
        
        #Ls the list of log likelihood for each iteration
        #including the initial one before optimization
        Ls = [Lnew,]
        #i would be iteraion index
        for i in range(max_iter):
            # calculate expected count for each component
            # given count specified by each set of indices
            # and add the count to corresponding factor 
            # matrices
            self.accumulate(indices, counts, sparse_update)
            # normalize the factor matrices so that they
            # can be interpreted as probabilities
            self.normalize()
            # assign new values to the parameters, reset
            # the temporary variables holding expected
            # count for each component
            self.update_value()
            
            # evaluating
            
            # calculate the new log likelihodd for this
            # iteration and store it in Ls
            
            # TODO: rewrite code to save some work
            #
            # we could get Log likelihood from accumulate 
            # when calling calculate_count_z, however, it
            # not Lnew, but L"old", as it is based on
            # parameter before update
            #
            # At first glance, it seems we could add 
            # computation of likelihood to method
            # calculate_count_z and pass probs_dict to
            # store the results
            #
            # But this is not a good idea, as the log
            # likelihood would be from the set of
            # parameters before the update. The log
            # likelihood from the current parameters
            # are needed to check early stopping criterion
            #
            # There are ways to circumvent this, but the
            # code would be not very intuitive
            #
            # Another approach is to make log likelihood
            # prepare count_z for next iteration, but this
            # is unintuitive too.
            #
            # Maybe what we could do is store intermediate
            # results from log likelihood somewhere. When
            # calculating anything, we can check if any
            # old intermediate result is available, and
            # when we update the parameters or change
            # dataset, we mark the intermediate results
            # expired. But this seems an overkill?
            
            # Could there be a better way?
            Lnew = self.get_log_likelihood(indices, counts)
            Ls.append(Lnew)
            # display the performance of current iteration
            if verbose:
                print("iteration "+str(i)+" training likelihood: "+str(np.round(Lnew,2)))
            
            # check stopping criterion
            
            # if the diff in log likelihood is too small
            # stop early
            diff = Ls[-1] - Ls[-2]
            #print(diff, Ls[0])
            if diff < tol:
                break
        # return the log likelihoods from each iteration
        return Ls
    def optimize_stop_by_valid_likelihood(self, indices, counts, max_iter, indices_valid, counts_valid, \
                                          N_decrs, back_up, sparse_update, verbose):
        """
        optimize model parameter using EM algorithm and tensor
        stored sparsely in the form of indices and counts
        
        optimization stops when the validation set log likelihood
        keep decreasing for more than threshold provided as
        N_decrs
        
        Parameters
        ----------
        indices : numpy array, of shape (N,dim) (see below)
            one of the two variables used to sparsely store tensor
            used to fit model
            the other variable is counts
            indices for the corresponding counts
            we store a count tensor sparsely by agreeing that element
            at index indices[i] has value (or count) counts[i]
            In this case, N above would be the number of non-zero
            elements, dim would be the number of different axis for
            tensor (i.e., the number of integers required to locate
            an element; for a matrix, dim=2)
        
        counts : numpy array, of shape (N,) (see below)
            one of the two variables used to sparsely store tensor
            used to fit model
            the other variable is indices
            counts for the corresponding indices
            we store a count tensor sparsely by agreeing that element
            at index indices[i] has value (or count) counts[i]
            In this case, N above would be the number of non-zero
            elements.
            
        max_iter : int, default = 300, value should be positive
            The maximum number of iterations to run.
            If reached, the optimization is stopped even if the stopping criterion
            if not met
        
        indices_valid : numpy array, of shape (N,dim) (see below)
            one of the two variables used to sparsely store tensor
            used to validate model
            the other being counts_valid
            indices for the corresponding count
            we store a count tensor sparsely by agreeing that element
            at index indices[i] has value (or count) counts[i]
            In this case, N above would be the number of non-zero
            elements, dim would be the number of different axis for
            tensor (i.e., the number of integers required to locate
            an element; for a matrix, dim=2)
        
        counts_valid : numpy array, of shape (N,) (see below)
            one of the two variables used to sparsely store tensor
            used to validate model
            the other being indices_valid
            counts for the corresponding indices
            we store a count tensor sparsely by agreeing that element
            at index indices[i] has value (or count) counts[i]
            In this case, N above would be the number of non-zero
            elements.
        
        N_decrs : int, value should be positive
            the maximum number of consecutive decrease in validation set log
            likelihood
            if exceeded, the optimization would stop
            only used if stop_criterion is "valid"
            
        back_up : bool
            whether to store the best set of parameters we have seen
            at each iteration
            if True,  the set of parameters with best validation set
            log likelihood would be returned, but additional resource
            would be required to store the best parameters
            if False, the last set of parameters would be returned,
            which has sub-optimal performance on validation set, but
            the optimization would run faster
            
        sparse_update : bool, default = True
            whether to use sparse update for adding counts to the
            factor matrices
            if True, update element by element. This is recommended
            when the factor matrices are expected to be sparse, as
            it would update less elements
            if False, update by constructing a new matrix for each
            factor matrix. This is recommended if the factor matrices
            are expected to be dense, as it can vectorization to
            speed up the ccomputation
            
        verbose : bool
            whether to enable screen output
            if so, log likelihood at each iteration would be displayed
            in addition to how the optimization stopped 
            (from reaching stopping criterion or exceeding limit on iteration)
            
        Returns
        -------
        Ls : list 
            a list of log likelihood at each iteration
            the type of log likelihood (from train or valid) depends on 
            stop_criterion
            the first element is the log likelihood from initial parameter without
            fitting
        """
        # calculate initial log likelihood
        
        #Lnew would be used for the new log likelihood
        #i.e. after the values are properly updated, for each iteration
        Lnew = self.get_log_likelihood(indices_valid, counts_valid)
        if verbose:
            print("initial training likelihood: ", np.round(Lnew,2))
        
        # optimizing
        
        #Ls the list of log likelihood for each iteration
        #including the initial one before optimization
        Ls = [Lnew,]
        #bestLL stores the best log likelihood we have so
        #far, so that we can keep track of best model
        bestLL = - np.inf
        #n_decrs : number of decrease in validation set log 
        #likelihood we have seen
        n_decrs = 0
        for i in range(max_iter):
            # calculate expected count for each component
            # given count specified by each set of indices
            # and add the count to corresponding factor 
            # matrices
            self.accumulate(indices, counts, sparse_update)
            # normalize the factor matrices so that they
            # can be interpreted as probabilities
            self.normalize()
            # assign new values to the parameters, reset
            # the temporary variables holding expected
            # count for each component
            self.update_value()
            
            # evaluating
            
            # calculate the new log likelihodd for this
            # iteration and store it in Ls
            Lnew = self.get_log_likelihood(indices_valid, counts_valid)
            Ls.append(Lnew)
            # display the performance of current iteration
            if verbose:
                print("iteration "+str(i)+" training likelihood: "+str(np.round(Lnew,2)))
                
            # check early stopping criterion
            
            #if the log likelihood decreased
            if Ls[-1] < Ls[-2]:
                #update the count of consecutive decrease
                n_decrs += 1
                #if the decrease we had is more than N_decrs
                if n_decrs > N_decrs:
                    #if we choose to store the best model
                    #we have seen as a back up
                    #restore it
                    if back_up:
                        self.p_z = best_p_z
                        self.factors = best_factors
                    #and stop optimization
                    break
            #if the log likelihood increased
            else:
                #reset the count of consecutive decrease
                n_decrs = 0
                #check if this is the best model so far
                #we only have to check when it is increasing
                #if decreasing the previous one is already 
                #better
                if back_up and Lnew > bestLL:
                    bestLL = Lnew
                    best_p_z = self.p_z.copy()
                    best_factors = deepcopy(self.factors)
        
        return Ls
    def accumulate(self, indices, counts, sparse_update = True):
        """
        Calculate the expected count for each outcome and component z (E-step)
        and add to the factor matrices (M-step)
        
        Parameters
        ----------
        indices : numpy array, of shape (N,dim) (see below)
            one of the two variables used to sparsely store tensor
            used to fit model
            the other variable is counts
            indices for the corresponding counts
            we store a count tensor sparsely by agreeing that element
            at index indices[i] has value (or count) counts[i]
            In this case, N above would be the number of non-zero
            elements, dim would be the number of different axis for
            tensor (i.e., the number of integers required to locate
            an element; for a matrix, dim=2)
        
        counts : numpy array, of shape (N,) (see below)
            one of the two variables used to sparsely store tensor
            used to fit model
            the other variable is indices
            counts for the corresponding indices
            we store a count tensor sparsely by agreeing that element
            at index indices[i] has value (or count) counts[i]
            In this case, N above would be the number of non-zero
            elements.
            
        sparse_update : bool, default = True
            whether to use sparse update for adding counts to the
            factor matrices
            if True, update element by element. This is recommended
            when the factor matrices are expected to be sparse, as
            it would update less elements
            if False, update by constructing a new matrix for each
            factor matrix. This is recommended if the factor matrices
            are expected to be dense, as it can vectorization to
            speed up the computation
            
        Returns
        -------
        Nothing
        
        """
        chunk_size = len(indices)
        #n_block = int((chunk_size-1)//block_size) + 1
        count_z = self.calculate_count_attributable_to_z(indices, counts)
        #it is tempting to update with vectorization
        #but it doesn't work, the same element in factor matrix
        #may appear multiple times on LHS, then only one of the
        #summand would be added to each element, resulting in
        #wrong results
        #for b in range(n_block):
        #    s = b*block_size
        #    e = (b+1)*block_size
        #    for d in range(self.dim):
        #        self.new_factors[d][indices[s:e,d]] += update[s:e]
        #    self.new_Pz += update[s:e].sum(axis=0)
        if sparse_update:
            n_block = int( (chunk_size-1) // self.block_size ) + 1
            for b in range(n_block):
                s = b*self.block_size
                e = s + self.block_size
                for d in range(self.dim):
                    outcome, pos = np.unique(indices[s:e,d], return_inverse = True)
                    for c in range(self.nc):
                        self.new_factors[d][outcome,c] += np.bincount(pos, weights = count_z[s:e,c])
                self.new_p_z += count_z[s:e].sum(axis=0)
            #for d in range(self.dim):
            #    outcome, pos = np.unique(indices[:,d], return_inverse = True)
            #    for c in range(self.nc):
            #        self.new_factors[d][outcome,c] += np.bincount(pos, weights = count_z[:,c])
            #self.new_p_z += count_z.sum(axis=0)
        else:
            for c in range(self.nc):
                for d in range(self.dim):
                    self.new_factors[d][:,c] += np.bincount(indices[:,d], weights = count_z[:,c], \
                                                            minlength = self.shape[d])
                self.new_p_z[c] += count_z[:,c].sum()
        
    def normalize(self):
        """
        Normalize factor matrices and weight so that they could
        be interpreted as probabilities
        """
        for factor in self.new_factors:
            factor /= factor.sum(axis=0, keepdims=True) + self.eps
        self.new_p_z /= self.new_p_z.sum() + self.eps
    def update_value(self):
        """
        Update the values of parameters
        Assign the new values stored in the temporary variables
        to the parameters, and reset the temporary variables to
        0 for the next iteration
        """
        for d in range(self.dim):
            self.factors[d][:] = self.new_factors[d]
        for d in range(self.dim):
            self.new_factors[d][:] = 0
        self.p_z[:] = self.new_p_z
        self.new_p_z[:] = 0
    def get_log_likelihood_dumb(self, T):
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
            LL = np.sum(T*np.log(tl.cp_to_tensor([self.p_z,self.factors])))
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
        indices : numpy array, of shape (N_unique,dim) (see below)
            one of the two variables used to sparsely store tensor
            sampled from model
            the other variable is counts
            indices for the corresponding counts
            we store a count tensor sparsely by agreeing that element
            at index indices[i] has value (or count) counts[i]
            In this case, N_unique above would be the number of 
            non-zero elements, in other words the number of unique
            outcomes in the sampled data. dim would be the number of 
            different axis for tensor (i.e., the number of integers 
            required to locate an element; for a matrix, dim=2)
        
        counts : numpy array, of shape (N_unique,) (see below)
            one of the two variables used to sparsely store tensor
            used to fit model
            the other variable is indices
            counts for the corresponding indices
            we store a count tensor sparsely by agreeing that element
            at index indices[i] has value (or count) counts[i]
            In this case, N_unique above would be the number of 
            non-zero elements, in other words the number of unique
            outcomes in the sampled data.
        """
        if self.params_set:
            
            # sample events
            
            #indices_list stores a list of numpy arrays
            #each array is 2d, representing outcomes
            #sampled from a component
            indices_list = []
            #component_counts stores the number of samples
            #from each component
            component_counts = np.random.multinomial(N, self.p_z)
            #for each component
            for c in range(self.nc):
                #if there is no sample from that component
                if component_counts[c] == 0:
                    #skip
                    continue
                #indices_tmp stores outcome from this component c
                indices_tmp = np.zeros((component_counts[c], self.dim), dtype = int)
                #each dimension (i.e. random variable) is sampled
                #independently
                for d in range(self.dim):
                    indices_tmp[:,d] = np.random.choice(self.shape[d], size = component_counts[c], \
                                                        p = self.factors[d][:,c])
                #store samples from this component in indices_list
                indices_list.append(indices_tmp)
                
            # count each type of event
            
            #to facilitate counting, we turn each set of indices
            #into corresponding 1d index in the flatten version
            #of the same array
            #for example, for a 2 by 2 matrix a, a[0,0], a[0,1],
            #a[1,0], a[1,1] would have 1d index 0,1,2,3
            #For a tensor T of shape Ni * Nj * Nk, T[i,j,k] has
            #1d index i*Nj*Nk+j*Nk+k
            #the i-th element of increment stores how much the
            #1d index increases if the i-th index increase by
            #1. For matrix a, it would be [2,1], for tensor T
            #it would be [Nj*Nk,Nk,1]
            increment = np.ones(self.dim, dtype=int)
            increment[:-1] = np.cumprod(self.shape[:0:-1])[::-1]
            #indices are the events in terms of positions in tensor
            indices = np.vstack(indices_list)
            #indices_1d are the corresponding 1d indices
            indices_1d = indices.dot(increment)
            #get the unique indices, the first appearance, and
            #where else they appear
            _, row_num, pos = np.unique(indices_1d, return_index = True, return_inverse = True)
            #count how many times each event (or set of indices)
            #appear
            counts = np.bincount(pos)
            return indices[row_num], counts
        else:
            raise ValueError("parameters not set")

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
