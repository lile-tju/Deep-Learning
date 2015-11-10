"""This tutorial introduces restricted boltzmann machines (RBM) using Theano.

Boltzmann Machines (BMs) are a particular form of energy-based model which
contain hidden variables. Restricted Boltzmann Machines further restrict BMs
to those without visible-visible and hidden-hidden connections.
"""
import time

try:
    import PIL.Image as Image
except ImportError:
    import Image

import numpy

import theano
import theano.tensor as T
import os

from theano.tensor.shared_randomstreams import RandomStreams

from utils import tile_raster_images
from logistic_sgd import load_data
from classScript import get_num_correct

THEANO_FLAGS='exception_verbosity = high'
# from convert import convert_yval

# theano.exception_verbosity = 
# theano.config.compute_test_value = 'raise'

# start-snippet-1
class classRBM(object):
    """Restricted Boltzmann Machine (RBM)  """
    def __init__(
        self,
        inputx=None,
        inputy=None,
        n_visible=784,
        n_hidden=500,
        n_output=10,
        U=None,
        outbias=None,
        W=None,
        hbias=None,
        vbias=None,
        numpy_rng=None,
        theano_rng=None,
        batch_size=20,
        testx=None,
        testy=None,
        n_samples=None
    ):
        """
        RBM constructor. Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa),
        as well as for performing CD updates.

        :param input: None for standalone RBMs or symbolic variable if RBM is
        part of a larger graph.

        :param n_visible: number of visible units

        :param n_hidden: number of hidden units

        :param W: None for standalone RBMs or symbolic variable pointing to a
        shared weight matrix in case RBM is part of a DBN network; in a DBN,
        the weights are shared between RBMs and layers of a MLP

        :param hbias: None for standalone RBMs or symbolic variable pointing
        to a shared hidden units bias vector in case RBM is part of a
        different network

        :param vbias: None for standalone RBMs or a symbolic variable
        pointing to a shared visible units bias
        """

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_samples = n_samples

        if numpy_rng is None:
            # create a number generator
            numpy_rng = numpy.random.RandomState(1234)

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if W is None:
            # W is initialized with `initial_W` which is uniformely
            # sampled from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible)) the output of uniform if
            # converted using asarray to dtype theano.config.floatX so
            # that the code is runable on GPU
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            # theano shared variables for weights and biases
            W = theano.shared(value=initial_W, name='W', borrow=True)


        if U is None:
            # W is initialized with `initial_W` which is uniformely
            # sampled from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible)) the output of uniform if
            # converted using asarray to dtype theano.config.floatX so
            # that the code is runable on GPU
            initial_U = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_output)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_output)),
                    size=(n_hidden, n_output)
                ),
                dtype=theano.config.floatX
            )
            # theano shared variables for weights and biases
            U = theano.shared(value=initial_U, name='U', borrow=True)


        if hbias is None:
            # create shared variable for hidden units bias
            hbias = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='hbias',
                borrow=True
            )

        if vbias is None:
            # create shared variable for visible units bias
            vbias = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                name='vbias',
                borrow=True
            )

        if outbias is None:
            # create shared variable for visible units bias
            outbias = theano.shared(
                value=numpy.zeros(
                    n_output,
                    dtype=theano.config.floatX
                ),
                name='outbias',
                borrow=True
            )


        # initialize input layer for standalone RBM or layer0 of DBN
        self.inputx = inputx
        self.inputy = inputy
        
        if not inputx:
            self.inputx = T.matrix('inputx')
        if not inputy:
            self.inputy = T.matrix('inputy')

        self.testx = testx
        self.testy = testy
        
        if not testx:
            self.testx = T.matrix('testx')
        if not testy:
            self.testy = T.matrix('testy')


        self.W = W
        self.U = U

        self.hbias = hbias
        self.vbias = vbias
        self.outbias = outbias
        self.batch_size = batch_size


        self.theano_rng = theano_rng
        # **** WARNING: It is not a good idea to put things in this list
        # other than shared variables created in this function.
        self.params = [self.W, self.hbias, self.vbias, self.U, self.outbias]
        # end-snippet-1

# ##########################################################################################################24june 3:35pm


    def free_energy(self, v_sample, y_sample):
        ''' Function to compute the free energy '''
        # print 'in#########'
        # print y_sample.eval()
        wx_b = T.dot(v_sample, self.W) + self.hbias + T.dot(y_sample, self.U.T)

        vbias_term = T.dot(v_sample, self.vbias)
        outbias_term = T.dot(y_sample, self.outbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        # print 'out#####'
        # print vbias_term.eval() , outbias_term.eval(), hidden_term.eval()
        return -hidden_term - vbias_term - outbias_term

    def free_energy1(self):
        ''' Function to compute the free energy '''
        # print 'in#########'
        # print y_sample.eval()
        wx_b = T.dot(self.inputx, self.W) + self.hbias + T.dot(self.inputy, self.U.T)

        vbias_term = T.dot(self.inputx, self.vbias)
        outbias_term = T.dot(self.inputy, self.outbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        print 'out#####'
        # print vbias_term.eval() , outbias_term.eval(), hidden_term.eval()
        return -hidden_term - vbias_term - outbias_term

    def propup(self, vis, y):
        '''This function propagates the visible units activation upwards to
        the hidden units

        Note that we return also the pre-sigmoid activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''
        pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias + T.dot(y, self.U.T)

        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_vy(self, v0_sample, y0_sample):
        ''' This function infers state of hidden units given visible units '''
        # compute the activation of the hidden units given a sample of
        # the visibles
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample, y0_sample)
        # get a sample of the hiddens given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                             n=1, p=h1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        '''This function propagates the hidden units activation downwards to
        the visible units

        Note that we return also the pre_sigmoid_activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''
        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def propout(self, hid):
        pre_softmax_activation = T.dot(hid, self.U) + self.outbias
        softmax_activation = T.nnet.softmax(pre_softmax_activation)  

        return [pre_softmax_activation, softmax_activation]

    def sample_v_given_h(self, h0_sample):
        ''' This function infers state of visible units given hidden units '''
        # compute the activation of the visible given the hidden sample
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        # get a sample of the visible given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
                                             n=1, p=v1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def get_y_proc(self,y1_sample,numberOne,dumm):
    	y_proc = T.zeros_like(dumm,dtype='int32')

    	for i in range(0,self.batch_size):
    		y_proc = T.set_subtensor(y_proc[i,y1_sample[i]], numberOne)

    	return y_proc

    def sample_y_given_h(self, h0_sample):
        pre_softmax_activation, softmax_activation = self.propout(h0_sample)

        y1_sample = T.argmax(softmax_activation, axis=1)

        # y_proc = T.matrix('y_proc',dtype='int32')
        # numberOne = T.scalar('numberOne')
        # dumm = T.matrix('dumm')
        y_proc = self.get_y_proc(y1_sample,1,numpy.zeros((self.batch_size, self.n_output)))

        return [pre_softmax_activation, softmax_activation, y_proc]


    def get_y_array(self,dumm):
        y_proc = T.zeros_like(dumm,dtype='int32')
        return y_proc

    
    def y_given_x(self,numberOne,y_proc,i,j):

        # y_proc = self.get_y_array(numpy.zeros((n_test_samples,self.n_output)))
        # n_test_samples = n_samples.get_value()
        # for j in xrange(self.n_samples):
        #     for i in xrange(self.n_output):
        activation_term = T.dot(self.testx[j,],self.W) + self.hbias + self.U.T[i,]
        expterm = T.exp(activation_term)
        sumterm = expterm + numberOne
        producterm = T.prod(sumterm)
        # y_proc = T.set_subtensor(y_proc[i,y[i]-1], a)

        y_proc = T.set_subtensor(y_proc[j,i],producterm * T.exp(self.outbias[i]))
                # y_proc[j,i] = producterm * T.exp(self.outbias[i])

        return y_proc
        # return T.sum(T.neq(T.argmax(y_proc,axis=0)+numberOne,self.testy))

    # def get_num_correct(self, y_proc):
    #     return T.sum(T.neq(T.argmax(y_proc,axis=0),self.testy))        


    # print f([2,3,5],1,10,numpy.zeros((mini_batch, 10),dtype='int32'))
    
        # y_proc = T.zeros_like(dumm)

        # f = theano.function([y,numberOne,dumm],y_proc,on_unused_input='ignore')

        # f1 = theano.function([dumm],y_proc)

        # f1(numpy.zeros((self.batch_size, 10),dtype='int32'))

        # for i in range(0,self.batch_size):
        # 	y_proc = T.set_subtensor(y_proc[i,y1_sample[i]], numberOne)

        # f = theano.function([numberOne,y1_sample],y_proc)
        # 
        # f(1,y1_sample,dtype='int32'))


        	# c=y1_sample.shape

        # return [pre_softmax_activation, softmax_activation, y_proc]

    def gibbs_hvh(self, h0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the hidden state'''
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_softmax_y1, y1_mean, y1_sample = self.sample_y_given_h(h0_sample)

        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_vy(v1_sample,y1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_softmax_y1, y1_mean, y1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample, y0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible state'''
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_vy(v0_sample, y0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        pre_softmax_y1, y1_mean, y1_sample = self.sample_y_given_h(h1_sample)

        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_softmax_y1, y1_mean, y1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    # start-snippet-2
    def get_cost_updates(self, lr=0.1, persistent=None, k=1):
        """This functions implements one step of CD-k or PCD-k

        :param lr: learning rate used to train the RBM

        :param persistent: None for CD. For PCD, shared variable
            containing old state of Gibbs chain. This must be a shared
            variable of size (batch size, number of hidden units).

        :param k: number of Gibbs steps to do in CD-k/PCD-k

        Returns a proxy for the cost and the updates dictionary. The
        dictionary contains the update rules for weights and biases but
        also an update of the shared variable used to store the persistent
        chain, if one is used.

        """

        # compute positive phase
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_vy(self.inputx, self.inputy)

        # decide how to initialize persistent chain:
        # for CD, we use the newly generate hidden sample
        # for PCD, we initialize from the old state of the chain
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent
        # end-snippet-2
        # perform actual negative phase
        # in order to implement CD-k/PCD-k we need to scan over the
        # function that implements one gibbs step k times.
        # Read Theano tutorial on scan for more information :
        # http://deeplearning.net/software/theano/library/scan.html
        # the scan will return the entire Gibbs chain
        (
            [
                pre_sigmoid_nvs,nv_means,nv_samples,
                pre_softmax_nys,ny_means,ny_samples,
                pre_sigmoid_nhs,nh_means,nh_samples
            ],
            updates
        ) = theano.scan(
            self.gibbs_hvh,
            # the None are place holders, saying that
            # chain_start is the initial state corresponding to the
            # 6th output
            outputs_info=[None, None, None, None, None, None, None, None, chain_start],
            n_steps=k
        )
        # start-snippet-3
        # determine gradients on RBM parameters
        # note that we only need the sample at the end of the chain
        chain_endx = nv_samples[-1]
        chain_endy = ny_samples[-1]

        cost = T.mean(self.free_energy(self.inputx,self.inputy)) - T.mean(
            self.free_energy(chain_endx, chain_endy))

        # We must not compute the gradient through the gibbs sampling
        gparams = T.grad(cost, self.params, consider_constant=[chain_endx, chain_endy])
        # end-snippet-3 start-snippet-4
        # constructs the update dictionary
        for gparam, param in zip(gparams, self.params):
            # make sure that the learning rate is of the right dtype
            updates[param] = param - gparam * T.cast(
                lr,
                dtype=theano.config.floatX
            )
        if persistent:
            # Note that this works only if persistent is a shared variable
            updates[persistent] = nh_samples[-1]
            # pseudo-likelihood is a better proxy for PCD
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        else:
            # reconstruction cross-entropy is a better proxy for CD
            # print 'else#####'
            monitoring_cost = self.get_reconstruction_cost(updates,
                                                           pre_sigmoid_nvs[-1])

        # print 'ff######'
        # print updates.eval()
        return monitoring_cost, updates
        # return updates
        # end-snippet-4

    def get_pseudo_likelihood_cost(self, updates):
        """Stochastic approximation to the pseudo-likelihood"""

        # index of bit i in expression p(x_i | x_{\i})
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')

        # binarize the input image by rounding to nearest integer
        xi = T.round(self.input)

        # calculate free energy for the given bit configuration
        fe_xi = self.free_energy(xi)

        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
        # the result to xi_flip, instead of working in place on xi.
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

        # calculate free energy with bit flipped
        fe_xi_flip = self.free_energy(xi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip -
                                                            fe_xi)))

        # increment bit_i_idx % number as part of updates
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

        return cost

    def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
        """Approximation to the reconstruction error

        Note that this function requires the pre-sigmoid activation as
        input.  To understand why this is so you need to understand a
        bit about how Theano works. Whenever you compile a Theano
        function, the computational graph that you pass as input gets
        optimized for speed and stability.  This is done by changing
        several parts of the subgraphs with others.  One such
        optimization expresses terms of the form log(sigmoid(x)) in
        terms of softplus.  We need this optimization for the
        cross-entropy since sigmoid of numbers larger than 30. (or
        even less then that) turn to 1. and numbers smaller than
        -30. turn to 0 which in terms will force theano to compute
        log(0) and therefore we will get either -inf or NaN as
        cost. If the value is expressed in terms of softplus we do not
        get this undesirable behaviour. This optimization usually
        works fine, but here we have a special case. The sigmoid is
        applied inside the scan op, while the log is
        outside. Therefore Theano will only see log(scan(..)) instead
        of log(sigmoid(..)) and will not apply the wanted
        optimization. We can not go and replace the sigmoid in scan
        with something else also, because this only needs to be done
        on the last step. Therefore the easiest and more efficient way
        is to get also the pre-sigmoid activation as an output of
        scan, and apply both the log and sigmoid outside scan such
        that Theano can catch and optimize the expression.

        """

        cross_entropy = T.mean(
            T.sum(
                self.inputx * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
                (1 - self.inputx) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),
                axis=1
            )
        )

        return cross_entropy

    




def test_rbm(learning_rate=0.1, training_epochs=5,
             dataset='mnist.pkl.gz', batch_size=20,
             n_chains=20, n_samples=10, output_folder='newRBM_plots',
             n_hidden=625,n_output=10):
    """
    Demonstrate how to train and afterwards sample from it using Theano.

    This is demonstrated on MNIST.

    :param learning_rate: learning rate used for training the RBM

    :param training_epochs: number of epochs used for training

    :param dataset: path the the pickled dataset

    :param batch_size: size of a batch used to train the RBM

    :param n_chains: number of parallel Gibbs chains to be used for sampling

    :param n_samples: number of samples to plot for each chain

    """
    datasets = load_data(dataset,1)

    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[2]

    # train_set_y = convert_yval(train_set_y)
    # test_set_y = convert_yval(train_set_y)

    # print type(train_set_x)
    
    # print 'sadfsdf'
    # print train_set_x[0,].eval()
    # print test_set_y[0:10].eval()
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    
    number_of_test_samples = test_set_x.get_value(borrow=True).shape[0]
    print number_of_test_samples


    # print n_train_batches

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.matrix('y',dtype='int32')
    test_x = T.matrix('test_x')
    test_y = T.matrix('test_y',dtype='int32')


    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    # initialize storage for the persistent chain (state = hidden
    # layer of chain)
    persistent_chain = theano.shared(numpy.zeros((batch_size, n_hidden),
                                                 dtype=theano.config.floatX),
                                     borrow=True)

    # construct the RBM class
    rbm = classRBM(inputx=x,inputy=y, testx=test_x,testy=test_y, n_visible=28 * 28,
              n_hidden=n_hidden, n_output=n_output,numpy_rng=rng, theano_rng=theano_rng, batch_size=batch_size, n_samples=number_of_test_samples)

    # get the cost and the gradient corresponding to one step of CD-15
    # cost, updates = rbm.get_cost_updates(lr=learning_rate, k=15)
    cost, updates = rbm.get_cost_updates(lr=learning_rate, k=1)

    # energy = theano.function([],rbm.free_energy1())
    # print 'Energy = ',energy

    #################################
    #     Training the RBM          #
    #################################
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    # start-snippet-5
    # it is ok for a theano function to have no output
    # the purpose of train_rbm is solely to update the RBM parameters
    train_rbm = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        },
        name='train_rbm'
    )

    plotting_time = 0.
    start_time = time.clock()

    # go through training epochs
    for epoch in xrange(training_epochs):

        # go through the training set
        mean_cost = []
        print 'number of batches = ', n_train_batches
        print 'Epoch = ',epoch
        for batch_index in xrange(n_train_batches):
            if(batch_index % 500 == 0):
                print batch_index
            mean_cost += [train_rbm(batch_index)]
            # train_rbm(batch_index)
            # print 'epoch 1######'


        print 'Training epoch %d, cost is ' % epoch, numpy.mean(mean_cost)
        # print 'Training epoch %d, cost is ' % epoch


        # num_correct = get_num_correct(rbm.W.get_value(borrow=True),
        # rbm.U.get_value(borrow=True),rbm.hbias.get_value(borrow=True),rbm.outbias.get_value(borrow=True),
        # test_set_x.eval(),test_set_y.eval(),number_of_test_samples,n_output)
        # print 'Accuracy = ', float(num_correct)/number_of_test_samples

        # Plot filters after each training epoch
        plotting_start = time.clock()
        # Construct image from the weight matrix
        image = Image.fromarray(
            tile_raster_images(
                X=rbm.W.get_value(borrow=True).T,
                img_shape=(28, 28),
                tile_shape=(10, 10),
                tile_spacing=(1, 1)
            )
        )
        image.save('filters_at_epoch_%i.png' % epoch)

        image = Image.fromarray(
            tile_raster_images(
                X=rbm.U.get_value(borrow=True).T,
                img_shape=(25, 25),
                tile_shape=(1, 10),
                tile_spacing=(1, 1)
            )
        )
        image.save('U_filters_at_epoch_%i.png' % epoch)
        plotting_stop = time.clock()
        plotting_time += (plotting_stop - plotting_start)

    end_time = time.clock()

    pretraining_time = (end_time - start_time) - plotting_time

    print ('Training took %f minutes' % (pretraining_time / 60.))

    # print rbm.W.get_value(borrow=True).shape
    # print rbm.U.get_value(borrow=True).shape
    # print rbm.hbias.get_value(borrow=True).shape
    # print rbm.outbias.get_value(borrow=True)
    # print test_set_y.eval()
   
    # print numpy.dot(test_set_x[1,].eval(), rbm.W.get_value(borrow=True)) + rbm.hbias.get_value(borrow=True) + numpy.transpose(rbm.U.get_value(borrow=True))[1,]


    # end-snippet-5 start-snippet-6
    #################################
    #     Sampling from the RBM     #
    #################################
    # find out the number of test samples
    # y_proc = T.matrix('y_proc')
    # i = T.iscalar('i')
    # j = T.iscalar('j')
    # dumm = numpy.zeros((number_of_test_samples, n_output))

    # n_samples = T.scalar('n_samples')

    # y_proc = T.zeros_like(dumm,dtype='int32')

    # numberOne = T.scalar('numberOne')
    # # n_test_samples = T.scalar('n_test_samples')

    # y_proc = rbm.y_given_x(numberOne,y_proc,i,j)
    # func = theano.function([numberOne,y_proc,i,j],y_proc,on_unused_input='ignore')

    # for jj in xrange(number_of_test_samples):
    #     print jj
    #     for ii in xrange(n_output):
    #         dumm = func(1,dumm,ii,jj)
        # print numpy.argmax(dumm[jj,])

    # print dumm
    # num_correct = 0
    # y_proc1 = T.matrix('y_proc1')
    # num_correct = rbm.get_num_correct(y_proc1)

    # error_func = theano.function([y_proc1],num_correct)

    # num_correct = error_func(dumm)
    # print num_correct/number_of_test_samples

    # # pick random test examples, with which to initialize the persistent chain
    # test_idx = rng.randint(number_of_test_samples - n_chains)
    # persistent_vis_chain = theano.shared(
    #     numpy.asarray(
    #         test_set_x.get_value(borrow=True)[test_idx:test_idx + n_chains],
    #         dtype=theano.config.floatX
    #     )
    # )
    # # end-snippet-6 start-snippet-7
    # plot_every = 1000
    # # define one step of Gibbs sampling (mf = mean-field) define a
    # # function that does `plot_every` steps before returning the
    # # sample for plotting
    # (
    #     [
    #         presig_hids,
    #         hid_mfs,
    #         hid_samples,
    #         presig_vis,
    #         vis_mfs,
    #         vis_samples
    #     ],
    #     updates
    # ) = theano.scan(
    #     rbm.gibbs_vhv,
    #     outputs_info=[None, None, None, None, None, persistent_vis_chain],
    #     n_steps=plot_every
    # )

    # # add to updates the shared variable that takes care of our persistent
    # # chain :.
    # updates.update({persistent_vis_chain: vis_samples[-1]})
    # # construct the function that implements our persistent chain.
    # # we generate the "mean field" activations for plotting and the actual
    # # samples for reinitializing the state of our persistent chain
    # sample_fn = theano.function(
    #     [],
    #     [
    #         vis_mfs[-1],
    #         vis_samples[-1]
    #     ],
    #     updates=updates,
    #     name='sample_fn'
    # )

    # # create a space to store the image for plotting ( we need to leave
    # # room for the tile_spacing as well)
    # image_data = numpy.zeros(
    #     (29 * n_samples + 1, 29 * n_chains - 1),
    #     dtype='uint8'
    # )
    # for idx in xrange(n_samples):
    #     # generate `plot_every` intermediate samples that we discard,
    #     # because successive samples in the chain are too correlated
    #     vis_mf, vis_sample = sample_fn()
    #     print ' ... plotting sample ', idx
    #     image_data[29 * idx:29 * idx + 28, :] = tile_raster_images(
    #         X=vis_mf,
    #         img_shape=(28, 28),
    #         tile_shape=(1, n_chains),
    #         tile_spacing=(1, 1)
    #     )

    # # construct image
    # image = Image.fromarray(image_data)
    # image.save('samples.png')
    # # end-snippet-7
    # os.chdir('../')

if __name__ == '__main__':
    test_rbm()
