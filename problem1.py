import numpy as np
import math 

#-------------------------------------------------------------------------
'''
    Problem 1: softmax regression 
    In this problem, you will implement the softmax regression for multi-class classification problems.
    The main goal of this problem is to extend the logistic regression method to solving multi-class classification problems.
    We will get familiar with computing gradients of vectors/matrices.
    We will use multi-class cross entropy as the loss function and stochastic gradient descent to train the model parameters.
    You could test the correctness of your code by typing `pytest -v test1.py` in the terminal.

    Notations:
            ---------- input data ----------------------
            p: the number of input features, an integer scalar.
            c: the number of classes in the classification task, an integer scalar.
            x: the feature vector of a data instance, a float numpy array of shape (p, ). 
            y: the label of a training instance, an integer scalar value. The values can be 0,1,2, ..., or (c-1).

            ---------- model parameters ----------------------
            W: the weight matrix of softmax regression, a float numpy array of shape (c, p). 
            b: the bias values of softmax regression, a float numpy array of shape (c, ).
            ---------- values ----------------------
            z: the linear logits, a float numpy array of shape (c, ).
            a: the softmax activations, a float numpy array of shape (c, ). 
            L: the multi-class cross entropy loss, a float scalar.

            ---------- partial gradients ----------------------
            dL_da: the partial gradients of the loss function L w.r.t. the activations a, a float numpy array of shape (c, ). 
                   The i-th element dL_da[i] represents the partial gradient of the loss function L w.r.t. the i-th activation a[i]:  d_L / d_a[i].
            da_dz: the partial gradient of the activations a w.r.t. the logits z, a float numpy array of shape (c, c). 
                   The (i,j)-th element of da_dz represents the partial gradient ( d_a[i]  / d_z[j] )
            dz_dW: the partial gradient of logits z w.r.t. the weight matrix W, a float numpy array of shape (c, p). 
                   The (i,j)-th element of dz_dW represents the partial gradient of the i-th logit (z[i]) w.r.t. the weight W[i,j]:   d_z[i] / d_W[i,j]
            dz_db: the partial gradient of the logits z w.r.t. the biases b, a float matrix of shape (c, ). 
                   Each element dz_db[i] represents the partial gradient of the i-th logit z[i] w.r.t. the i-th bias b[i]:  d_z[i] / d_b[i]

            ---------- partial gradients of parameters ------------------
            dL_dW: the partial gradients of the loss function L w.r.t. the weight matrix W, a float numpy array of shape (c, p). 
                   The i,j-th element dL_dW[i,j] represents the partial gradient of the loss function L w.r.t. the i,j-th weight W[i,j]:  d_L / d_W[i,j]
            dL_db: the partial gradient of the loss function L w.r.t. the biases b, a float numpy array of shape (c, ).
                   The i-th element dL_db[i] represents the partial gradient of the loss function w.r.t. the i-th bias:  d_L / d_b[i]

            ---------- training ----------------------
            alpha: the step-size parameter of gradient descent, a float scalar.
            n_epoch: the number of passes to go through the training dataset in order to train the model, an integer scalar.
'''

#-----------------------------------------------------------------
# Forward Pass 
#-----------------------------------------------------------------

#-----------------------------------------------------------------
def compute_z(x, W, b):
    z = W @ x + b
    return z



#-----------------------------------------------------------------
def compute_a(z):
    z = np.asarray(z, dtype=np.float64).ravel()          # copy & flatten
    shifted = z - np.max(z)                              # stability shift
    with np.errstate(under='ignore'):                    # silence exp under-flow
        exp_z = np.exp(shifted)
    return exp_z / exp_z.sum()



#-----------------------------------------------------------------
def compute_L(a, y):
    """Single-example cross-entropy."""
    prob = float(np.asarray(a).ravel()[y])
    return float('inf') if prob == 0.0 else -math.log(prob)




#-----------------------------------------------------------------
def forward(x, y, W, b):
    z = compute_z(x, W, b)
    a = compute_a(z)
    L = compute_L(a, y)
    return z, a, L



#-----------------------------------------------------------------
# Compute Local Gradients
#-----------------------------------------------------------------



#-----------------------------------------------------------------
def compute_dL_da(a, y):
    a = np.asarray(a).reshape(-1)
    dL_da = np.zeros_like(a)
    dL_da[y] = -1e10 if a[y] == 0.0 else -1.0 / a[y]
    return dL_da




#-----------------------------------------------------------------
def compute_da_dz(a):
    c = a.shape[0]
    da_dz = np.zeros((c, c))
    for i in range(c):
        for j in range(c):
            if i == j:
                da_dz[i, j] = a[i] * (1 - a[i])
            else:
                da_dz[i, j] = -a[i] * a[j]
    return da_dz


#-----------------------------------------------------------------
def compute_dz_dW(x, c):
    x = np.asarray(x).reshape(-1)                   # shape → (p,)
    return np.tile(x, (c, 1))




#-----------------------------------------------------------------
def compute_dz_db(c):
    return np.ones(c)


#-----------------------------------------------------------------
# Back Propagation 
#-----------------------------------------------------------------

#-----------------------------------------------------------------
def backward(x, y, a):
    dL_da = compute_dL_da(a, y)
    da_dz = compute_da_dz(a)
    dz_dW = compute_dz_dW(x, len(a))
    dz_db = compute_dz_db(len(a))
    return dL_da, da_dz, dz_dW, dz_db


#-----------------------------------------------------------------
def compute_dL_dz(dL_da, da_dz):
    dL_dz = da_dz.T @ dL_da
    return dL_dz



#-----------------------------------------------------------------
def compute_dL_dW(dL_dz, dz_dW):
    dL_dW = np.zeros_like(dz_dW)
    for i in range(len(dL_dz)):
        dL_dW[i, :] = dL_dz[i] * dz_dW[i, :]
    return dL_dW



#-----------------------------------------------------------------
def compute_dL_db(dL_dz, dz_db):
    dL_db = dL_dz * dz_db
    return dL_db

#-----------------------------------------------------------------
# gradient descent 
#-----------------------------------------------------------------

#--------------------------
def update_W(W, dL_dW, alpha=0.001):
    W = W - alpha * dL_dW
    return W



#--------------------------
def update_b(b, dL_db, alpha=0.001):
    b = b - alpha * dL_db
    return b

#--------------------------
# train
def train(X, Y, alpha=0.01, n_epoch=100):
    p = X.shape[1]
    c = max(Y) + 1

    W = np.random.rand(c, p)
    b = np.random.rand(c)

    for _ in range(n_epoch):
        for x, y in zip(X, Y):
            z, a, L = forward(x, y, W, b)
            dL_da, da_dz, dz_dW, dz_db = backward(x, y, a)
            dL_dz = compute_dL_dz(dL_da, da_dz)
            dL_dW = compute_dL_dW(dL_dz, dz_dW)
            dL_db = compute_dL_db(dL_dz, dz_db)
            W = update_W(W, dL_dW, alpha)
            b = update_b(b, dL_db, alpha)
    return W, b


#--------------------------
def predict(Xtest, W, b):
    n = Xtest.shape[0]
    c = W.shape[0]
    Y = np.zeros(n, dtype=int)
    P = np.zeros((n, c))
    for i, x in enumerate(Xtest):
        z = compute_z(x, W, b)
        a = compute_a(z)
        Y[i] = np.argmax(a)
        P[i] = a
    return Y, P




#-----------------------------------------------------------------
# gradient checking 
#-----------------------------------------------------------------


#-----------------------------------------------------------------
def check_da_dz(z, delta=1e-7):
    '''
        Compute local gradient of the softmax function using gradient checking.
        Input:
            z: the logit values of softmax regression, a float numpy vector of shape (c, ). Here c is the number of classes
            delta: a small number for gradient check, a float scalar.
        Output:
            da_dz: the approximated local gradient of the activations w.r.t. the logits, a float numpy array of shape (c, c). 
                   The (i,j)-th element represents the partial gradient ( d a[i]  / d z[j] )
    '''
    c = z.shape[0] # number of classes
    da_dz = np.zeros((c, c))
    for i in range(c):
        for j in range(c):
            d = np.zeros(c)
            d[j] = delta
            da_dz[i, j] = (compute_a(z + d)[i] - compute_a(z)[i]) / delta
    return da_dz 

#-----------------------------------------------------------------
def check_dL_da(a, y, delta=1e-7):
    '''
        Compute local gradient of the multi-class cross-entropy function w.r.t. the activations using gradient checking.
        Input:
            a: the activations of a training instance, a float numpy vector of shape (c, ). Here c is the number of classes. 
            y: the label of a training instance, an integer scalar value. The values can be 0,1,2, ..., or (c-1).
            delta: a small number for gradient check, a float scalar.
        Output:
            dL_da: the approximated local gradients of the loss function w.r.t. the activations, a float numpy vector of shape (c, ).
    '''
    c = a.shape[0] # number of classes
    dL_da = np.zeros(c) # initialize the vector as all zeros
    #print(dL_da)
    for i in range(c):
        d = np.zeros(c)
        d[i] = delta
        dL_da[i] = (compute_L(a + d, y) - compute_L(a, y)) / delta
    return dL_da 

#--------------------------
def check_dz_dW(x, W, b, delta=1e-7):
    '''
        compute the local gradient of the logit function using gradient check.
        Input:
            x: the feature vector of a data instance, a float numpy vector of shape (p, ). Here p is the number of features/dimensions.
            W: the weight matrix of softmax regression, a float numpy array of shape (c, p). Here c is the number of classes.
            b: the bias values of softmax regression, a float numpy vector of shape (c, ).
            delta: a small number for gradient check, a float scalar.
        Output:
            dz_dW: the approximated local gradient of the logits w.r.t. the weight matrix computed by gradient checking, a float numpy array of shape (c, p). 
                   The i,j -th element of dz_dW represents the partial gradient of the i-th logit (z[i]) w.r.t. the weight W[i,j]:   d_z[i] / d_W[i,j]
    '''
    c, p = W.shape # number of classes and features
    dz_dW = np.zeros((c, p))
    for i in range(c):
        for j in range(p):
            d = np.zeros((c, p))
            d[i, j] = delta
            dz_dW[i, j] = (compute_z(x, W + d, b)[i] - compute_z(x, W, b))[i] / delta
    return dz_dW


#--------------------------
def check_dz_db(x, W, b, delta=1e-7):
    '''
        compute the local gradient of the logit function using gradient check.
        Input:
            x: the feature vector of a data instance, a float numpy vector of shape (p, ). Here p is the number of features/dimensions.
            W: the weight matrix of softmax regression, a float numpy array of shape (c, p). Here c is the number of classes.
            b: the bias values of softmax regression, a float numpy vector of shape (c, ).
            delta: a small number for gradient check, a float scalar.
        Output:
            dz_db: the approximated local gradient of the logits w.r.t. the biases using gradient check, a float vector of shape (c, ).
                   Each element dz_db[i] represents the partial gradient of the i-th logit z[i] w.r.t. the i-th bias:  d_z[i] / d_b[i]
    '''
    c, _ = W.shape # number of classes and features
    dz_db = np.zeros(c)
    for i in range(c):
        d = np.zeros(c)
        d[i] = delta
        dz_db[i] = (compute_z(x, W, b + d)[i] - compute_z(x, W, b)[i]) / delta
    return dz_db


#-----------------------------------------------------------------
def check_dL_dW(x,y,W,b,delta=1e-7):
    '''
       Compute the gradient of the loss function w.r.t. the weights W using gradient checking.
        Input:
            x: the feature vector of a training instance, a float numpy vector of shape (p, ). Here p is the number of features/dimensions.
            y: the label of a training instance, an integer scalar value. The values can be 0,1,2, ..., or (c-1).
            W: the weight matrix of softmax regression, a float numpy array of shape (c, p). Here c is the number of classes.
            b: the bias values of softmax regression, a float numpy vector of shape (c, ).
            delta: a small number for gradient check, a float scalar.
        Output:
            dL_dW: the approximated gradients of the loss function w.r.t. the weight matrix, a float numpy array of shape (c, p). 
    '''
    c, p = W.shape    
    dL_dW = np.zeros((c, p))
    for i in range(c):
        for j in range(p):
            d = np.zeros((c, p))
            d[i, j] = delta
            dL_dW[i, j] = (forward(x, y, W + d, b)[-1] - forward(x, y, W, b)[-1]) / delta
    return dL_dW


#-----------------------------------------------------------------
def check_dL_db(x,y,W,b,delta=1e-7):
    '''
       Compute the gradient of the loss function w.r.t. the bias b using gradient checking.
        Input:
            x: the feature vector of a training instance, a float numpy vector of shape (p, ). Here p is the number of features/dimensions.
            y: the label of a training instance, an integer scalar value. The values can be 0,1,2, ..., or (c-1).
            W: the weight matrix of softmax regression, a float numpy array of shape (c, p). Here c is the number of classes.
            b: the bias values of softmax regression, a float numpy vector of shape (c, ).
            delta: a small number for gradient check, a float scalar.
        Output:
            dL_db: the approxmiated gradients of the loss function w.r.t. the biases, a float vector of shape (c, ).
    '''
    c, _ = W.shape
    dL_db = np.zeros(c)
    for i in range(c):
        d = np.zeros(c)
        d[i] = delta
        loss_plus  = forward(x, y, W, b + d)[-1]
        loss_minus = forward(x, y, W, b - d)[-1]
        dL_db[i] = (loss_plus - loss_minus) / (2 * delta)
    return dL_db
