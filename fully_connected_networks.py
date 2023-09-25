"""
Implements fully connected networks in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cpu()" in each implementation block.
"""
import torch
from a2_helper import softmax_loss
from cs6476 import Solver


def hello_fully_connected_networks():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Hello from fully_connected_neural_networks.py!!!!!!")


class Linear(object):
    @staticmethod
    def forward(x, w, b):
        """
        Computes the forward pass for an linear (fully-connected) layer.
        The input x has shape (N, D) where N is the mini batch size and D is the dimension of
        the flattened image vector.
        Inputs:
        - x: A tensor containing input data, of shape (N, D)
        - w: A tensor of weights, of shape (D, M)
        - b: A tensor of biases, of shape (M,)
        Returns a tuple of:
        - out: output, of shape (N, M)
        - cache: (x, w, b)
        """
        out = None
        ######################################################################
        # TODO: Implement the linear forward pass. Store the result in out.  #
        # You will need to reshape the input into rows.                      #
        ######################################################################
        # x = x.reshape(x.shape[0], -1)
        out = torch.mm(x.reshape(x.shape[0], -1), w) + b
        ######################################################################
        #                        END OF YOUR CODE                            #
        ######################################################################
        cache = (x, w, b)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for an linear layer.
        Inputs:
        - dout: Upstream derivative, of shape (N, M)
        - cache: Tuple of:
          - x: Input data, of shape (N, D)
          - w: Weights, of shape (D, M)
          - b: Biases, of shape (M,)
        Returns a tuple of:
        - dx: Gradient with respect to x, of shape
          (N, D)
        - dw: Gradient with respect to w, of shape (D, M)
        - db: Gradient with respect to b, of shape (M,)
        """
        x, w, b = cache

        dx, dw, db = None, None, None
        db = dout.sum(dim=0)
        dx = dout.mm(w.t()).view(x.shape)
        dw = x.view(x.shape[0], -1).t().mm(dout)

        return dx, dw, db


class ReLU(object):
    @staticmethod
    def forward(x):
        """
        Computes the forward pass for a layer of rectified
        linear units (ReLUs).
        Input:
        - x: Input; a tensor of any shape
        Returns a tuple of:
        - out: Output, a tensor of the same shape as x
        - cache: x
        """
        out = None
        ###################################################
        # TODO: Implement the ReLU forward pass.          #
        # You should not change the input tensor with an  #
        # in-place operation.                             #
        ###################################################
        out = x.clone()
        out[out < 0] = 0

        ###################################################
        #                 END OF YOUR CODE                #
        ###################################################
        cache = x
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for a layer of rectified
        linear units (ReLUs).
        Input:
        - dout: Upstream derivatives, of any shape
        - cache: Input x, of same shape as dout
        Returns:
        - dx: Gradient with respect to x
        """
        dx, x = None, cache
        dx = dout.clone()
        dx[x < 0] = 0
        return dx


class Linear_ReLU(object):
    @staticmethod
    def forward(x, w, b):
        """
        Convenience layer that performs an linear transform
        followed by a ReLU.

        Inputs:
        - x: Input to the linear layer
        - w, b: Weights for the linear layer
        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Object to give to the backward pass
        """
        a, fc_cache = Linear.forward(x, w, b)
        out, relu_cache = ReLU.forward(a)
        cache = (fc_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the linear-relu convenience layer
        """
        fc_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = Linear.backward(da, fc_cache)
        return dx, dw, db


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.
    The architecure should be linear - relu - linear - softmax.
    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to PyTorch tensors.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=torch.float32,
        device="cpu",
    ):
        """
        Initialize a new network.
        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        - dtype: A torch data type object; all computations will be
          performed using this datatype. float is faster but less accurate,
          so you should use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cpu'
        """
        self.params = {}
        self.reg = reg

        ###################################################################
        # TODO: Initialize the weights and biases of the two-layer net.   #
        # Weights should be initialized from a Gaussian centered at       #
        # 0.0 with standard deviation equal to weight_scale, and biases   #
        # should be initialized to zero. All weights and biases should    #
        # be stored in the dictionary self.params, with first layer       #
        # weights and biases using the keys 'W1' and 'b1' and second layer#
        # weights and biases using the keys 'W2' and 'b2'.                #
        ###################################################################
        # Replace "pass" statement with your code
        self.params["W1"] = torch.normal(0, weight_scale, (input_dim, hidden_dim)).to(
            torch.float64
        )
        self.params["W2"] = torch.normal(0, weight_scale, (hidden_dim, num_classes)).to(
            torch.float64
        )
        self.params["b1"] = torch.zeros(hidden_dim).to(torch.float64)
        self.params["b2"] = torch.zeros(num_classes).to(torch.float64)

        ###############################################################
        #                            END OF YOUR CODE                 #
        ###############################################################

    def save(self, path):
        checkpoint = {
            "reg": self.reg,
            "params": self.params,
        }

        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location="cpu")
        self.params = checkpoint["params"]
        self.reg = checkpoint["reg"]
        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)
        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Tensor of input data of shape (N, d_1, ..., d_k)
        - y: int64 Tensor of labels, of shape (N,). y[i] gives the
          label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model
        and return:
        - scores: Tensor of shape (N, C) giving classification scores,
          where scores[i, c] is the classification score for X[i]
          and class c.
        If y is not None, then run a training-time forward and backward
        pass and return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping
          parameter names to gradients of the loss with respect to
          those parameters.
        """
        scores = None
        #############################################################
        # TODO: Implement the forward pass for the two-layer net,   #
        # computing the class scores for X and storing them in the  #
        # scores variable.                                          #
        #############################################################
        Linear_ReLU_out, cache_LR = Linear_ReLU.forward(
            X, self.params["W1"], self.params["b1"]
        )
        scores, cache_L = Linear.forward(
            Linear_ReLU_out, self.params["W2"], self.params["b2"]
        )

        ##############################################################
        #                     END OF YOUR CODE                       #
        ##############################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ###################################################################
        # TODO:                                                           #
        # Store the loss in the loss variable and gradients in the grads  #
        # dictionary. Compute data loss using softmax. Don't forget to    #                                      #
        # add L2 regularization!                                                                 #
        ###################################################################
        loss, dout = softmax_loss(scores, y)
        for i in range(1, 3):
            loss += (self.params[f"W{i}"] * self.params[f"W{i}"]).sum() * self.reg
        # print(f"y={y} \n scores={scores} \n loss={loss} \n dout={dout}")

        ###################################################################
        #                     END OF YOUR CODE                            #
        ###################################################################
        dx, dw, db = Linear.backward(dout, cache_L)
        grads["W2"] = dw + 2 * self.params["W2"] * self.reg
        grads["b2"] = db

        dx, dw, db = Linear_ReLU.backward(dx, cache_LR)
        grads["W1"] = dw + 2 * self.params["W1"] * self.reg
        grads["b1"] = db
        # dout is the final output besides loss from the softmax function
        # cache_L is the cache created in your forward pass from the final layer
        # cache_LR is the cache created from the forward pass of your first layer i.e linear ReLU layer

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function.
    For a network with L layers, the architecture will be:

    {linear - relu - [dropout]} x (L - 1) - linear - softmax

    where dropout is optional, and the {...} block is repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout=0.0,
        reg=0.0,
        weight_scale=1e-2,
        seed=None,
        dtype=torch.float,
        device="cpu",
    ):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving the drop probability for networks
          with dropout. If dropout=0 then the network should not use dropout.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        - dtype: A torch data type object; all computations will be performed using
          this datatype. float is faster but less accurate, so you should use
          double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.use_dropout = dropout != 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        ############################################################################
        # start layer
        self.params["W1"] = torch.normal(
            0, weight_scale, (input_dim, hidden_dims[0])
        ).to(torch.float64)
        self.params["b1"] = torch.zeros(hidden_dims[0]).to(torch.float64)

        for i in range(2, self.num_layers):
            self.params[f"W{i}"] = torch.normal(
                0, weight_scale, (hidden_dims[i - 2], hidden_dims[i - 1])
            ).to(torch.float64)
            self.params[f"b{i}"] = torch.zeros(hidden_dims[i - 1]).to(torch.float64)

        # last layer
        self.params[f"W{self.num_layers}"] = torch.normal(
            0, weight_scale, (hidden_dims[-1], num_classes)
        ).to(torch.float64)
        self.params[f"b{self.num_layers}"] = torch.zeros(num_classes).to(torch.float64)

        # for i in range(1, self.num_layers):
        #     print(self.params[f"W{i}"].shape)
        #     print(self.params[f"b{i}"].shape)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout}
            if seed is not None:
                self.dropout_param["seed"] = seed

    def save(self, path):
        checkpoint = {
            "reg": self.reg,
            "dtype": self.dtype,
            "params": self.params,
            "num_layers": self.num_layers,
            "use_dropout": self.use_dropout,
            "dropout_param": self.dropout_param,
        }

        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location="cpu")
        self.params = checkpoint["params"]
        self.dtype = dtype
        self.reg = checkpoint["reg"]
        self.num_layers = checkpoint["num_layers"]
        self.use_dropout = checkpoint["use_dropout"]
        self.dropout_param = checkpoint["dropout_param"]

        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)

        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.
        Input / output: Same as TwoLayerNet above.
        """
        # X = X.to(self.dtype)
        X = X.to(torch.float64)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        ############################################################################
        cache_dict = (
            {}
        )  # cache_dict is the dict saving the cache of output from each layer.

        for i in range(1, self.num_layers):
            X, cache_dict[f"cache_LR{i}"] = Linear_ReLU.forward(
                X, self.params[f"W{i}"], self.params[f"b{i}"]
            )
            if self.use_dropout:
                X, cache_dict[f"cache_Dropout{i}"] = Dropout.forward(
                    X, self.dropout_param
                )
        scores, cache_dict[f"cache_L{self.num_layers}"] = Linear.forward(
            X,
            self.params[f"W{self.num_layers}"],
            self.params[f"b{self.num_layers}"],
        )

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        i = self.num_layers
        loss, dout = softmax_loss(scores, y)
        loss += (
            self.params["W{}".format(i)] * self.params["W{}".format(i)]
        ).sum() * self.reg
        last_dout, dw, db = Linear.backward(dout, cache_dict["cache_L{}".format(i)])
        grads["W{}".format(i)] = dw + 2 * self.params["W{}".format(i)] * self.reg
        grads["b{}".format(i)] = db
        for n in range(self.num_layers - 1)[::-1]:
            i = n + 1
            if self.use_dropout:
                last_dout = Dropout.backward(
                    last_dout, cache_dict["cache_Dropout{}".format(i)]
                )
            last_dout, dw, db = Linear_ReLU.backward(
                last_dout, cache_dict["cache_LR{}".format(i)]
            )
            grads["W{}".format(i)] = dw + 2 * self.params["W{}".format(i)] * self.reg
            grads["b{}".format(i)] = db
            loss += (
                self.params["W{}".format(i)] * self.params["W{}".format(i)]
            ).sum() * self.reg
        # dout is the final output besides loss from the softmax function
        # cache_dict is the dict saving the cache of output from each layer.
        # cache_L{i} or cache_LR{i} is the key to save the cache of each layer in cronological order.
        # Hint: use for loop for the forward pass as well to accomodate arbitrary network depth.

        return loss, grads


def create_solver_instance(data_dict, dtype, device):
    model = TwoLayerNet(hidden_dim=200, dtype=dtype, device=device)
    solver = Solver(model, data_dict, device=device, num_epochs=100)
    return solver


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.
    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)

    w -= config["learning_rate"] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.
    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to
      store a moving average of the gradients.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("momentum", 0.9)
    v = config.get("velocity", torch.zeros_like(w))

    next_w = None
    ##################################################################
    # TODO: Implement the momentum update formula. Store the         #
    # updated value in the next_w variable. You should also use and  #
    # update the velocity v.                                         #
    ##################################################################
    v = config["momentum"] * v - config["learning_rate"] * dw
    next_w = w + v
    ###################################################################
    #                           END OF YOUR CODE                      #
    ###################################################################
    config["velocity"] = v

    return next_w, config


def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.
    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", torch.zeros_like(w))
    config.setdefault("v", torch.zeros_like(w))
    config.setdefault("t", 0)

    next_w = None
    config["t"] += 1
    config["m"] = config["beta1"] * config["m"] + (1 - config["beta1"]) * dw
    mt = config["m"] / (1 - config["beta1"] ** config["t"])
    config["v"] = config["beta2"] * config["v"] + (1 - config["beta2"]) * (dw * dw)
    vc = config["v"] / (1 - (config["beta2"] ** config["t"]))
    w = w - (config["learning_rate"] * mt) / (torch.sqrt(vc) + config["epsilon"])
    next_w = w

    return next_w, config


class Dropout(object):
    @staticmethod
    def forward(x, dropout_param):
        """
        Performs the forward pass for (inverted) dropout.
        Inputs:
        - x: Input data: tensor of any shape
        - dropout_param: A dictionary with the following keys:
          - p: Dropout parameter. We *drop* each neuron output with
            probability p.
          - mode: 'test' or 'train'. If the mode is train, then
            perform dropout;
          if the mode is test, then just return the input.
          - seed: Seed for the random number generator. Passing seed
            makes this
            function deterministic, which is needed for gradient checking
            but not in real networks.
        Outputs:
        - out: Tensor of the same shape as x.
        - cache: tuple (dropout_param, mask). In training mode, mask
          is the dropout mask that was used to multiply the input; in
          test mode, mask is None.
        NOTE: Please implement **inverted** dropout, not the vanilla
              version of dropout.
        See http://cs231n.github.io/neural-networks-2/#reg for more details.
        NOTE 2: Keep in mind that p is the probability of **dropping**
                a neuron output; this might be contrary to some sources,
                where it is referred to as the probability of keeping a
                neuron output.
        """
        p, mode = dropout_param["p"], dropout_param["mode"]
        if "seed" in dropout_param:
            torch.manual_seed(dropout_param["seed"])

        mask = None
        out = None

        if mode == "train":
            ##############################################################
            # TODO: Implement training phase forward pass for            #
            # inverted dropout.                                          #
            # Store the dropout mask in the mask variable.               #
            ##############################################################
            mask = torch.rand(x.shape) > p
            out = x * mask * (1 / (1 - p))

            ##############################################################
            #                   END OF YOUR CODE                         #
            ##############################################################
        elif mode == "test":
            ##############################################################
            # TODO: Implement the test phase forward pass for            #
            # inverted dropout.                                          #
            ##############################################################
            mask = torch.ones(x.shape)
            out = x * mask
            ##############################################################
            #                      END OF YOUR CODE                      #
            ##############################################################

        cache = (dropout_param, mask)

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Perform the backward pass for (inverted) dropout.
        Inputs:
        - dout: Upstream derivatives, of any shape
        - cache: (dropout_param, mask) from Dropout.forward.
        """
        dropout_param, mask = cache
        mode = dropout_param["mode"]

        dx = None
        if mode == "train":
            dx = dout / (1 - dropout_param["p"])
            dx[mask == 0] = 0
        elif mode == "test":
            dx = dout
        return dx


def optimal(data):
    ##############################################################
    # TODO: Optimal model                                        #
    #                                                            #
    ##############################################################
    # Replace "pass" statement with your code
    path = "fcn_model.pth"
    model = FullyConnectedNet([256], dropout=0.5, dtype=torch.float32, device="cpu")
    solver = Solver(
        model,
        data,
        num_epochs=100,
        batch_size=512,
        update_rule=sgd_momentum,
        optim_config={
            "learning_rate": 5e-3,
        },
        print_every=100000,
        print_acc_every=10,
        verbose=True,
        device="cpu",
    )
    solver.train()  # dont forget to comment this line in final submission after you have trained your model
    solver.model.save(
        path
    )  # dont forget to comment this line in final submision after you have saved your model
    return solver
    ##############################################################
    #                      END OF YOUR CODE                      #
    ##############################################################

    return 0


def gtid():
    ##############################################################
    # TODO: add correct gtid                                     #
    #                                                            #
    ##############################################################
    return 903522014
    ##############################################################
    #                      END OF YOUR CODE                      #
    ##############################################################
    return 0