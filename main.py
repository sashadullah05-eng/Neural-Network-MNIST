from tensorflow.keras.datasets import mnist 
import os
import numpy as np
import pandas as pd
from scipy.special import softmax as sfm #inbuild softmax function

# Initializing and getting Parameters
def rand_parameters() -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """
    Initializes random network parameters.
    Returns:
        Tuple of weight and bias matrices (W0, b0, W1, b1, W2, b2)
    """
    rng = np.random.default_rng()
    W0 = pd.DataFrame(rng.uniform(-0.5, 0.5, size=(32, 784)))
    b0 = pd.DataFrame(rng.uniform(-0.5, 0.5, size=(32, 1)))
    W1 = pd.DataFrame(rng.uniform(-0.5, 0.5, size=(32, 32)))
    b1 = pd.DataFrame(rng.uniform(-0.5, 0.5, size=(32, 1)))
    W2 = pd.DataFrame(rng.uniform(-0.5, 0.5, size=(10, 32)))
    b2 = pd.DataFrame(rng.uniform(-0.5, 0.5, size=(10, 1)))
    return W0, b0, W1, b1, W2, b2
def write_new_par() -> None:
    W0,b0,W1,b1,W2,b2=rand_parameters()
    try:
        write_par_to_excel(W0,b0,W1,b1,W2,b2) 
    except FileNotFoundError as e:
        with pd.ExcelWriter(r'Neural Network_Parameters.xlsx',mode='w') as writer:
            pd.DataFrame(W0).to_excel(writer,sheet_name='W0',header=False,index=False)
            pd.DataFrame(W1).to_excel(writer,sheet_name='W1',header=False,index=False)
            pd.DataFrame(W2).to_excel(writer,sheet_name='W2',header=False,index=False)
            pd.DataFrame(b0).to_excel(writer,sheet_name='b0',header=False,index=False)
            pd.DataFrame(b1).to_excel(writer,sheet_name='b1',header=False,index=False)
            pd.DataFrame(b2).to_excel(writer,sheet_name='b2',header=False,index=False)
def read_par() -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """
    Reads network parameters from Excel file.
    Returns:
        Tuple of weight and bias matrices (W0, b0, W1, b1, W2, b2)
    """
    if os.path.exists(r'Neural Network_Parameters.xlsx')==False:
        print("Parameter file not found.")
        print("Please initialize parameters first by running 'write_new_par()' function.")
        exit()
    with pd.ExcelFile(r'Neural Network_Parameters.xlsx') as xls:
        W0=np.array(pd.read_excel(xls,sheet_name='W0',header=None),dtype=float)
        W1=np.array(pd.read_excel(xls,sheet_name='W1',header=None),dtype=float)
        W2=np.array(pd.read_excel(xls,sheet_name='W2',header=None),dtype=float)
        b0=np.array(pd.read_excel(xls,sheet_name='b0',header=None),dtype=float)
        b1=np.array(pd.read_excel(xls,sheet_name='b1',header=None),dtype=float)
        b2=np.array(pd.read_excel(xls,sheet_name='b2',header=None),dtype=float)
    return W0,b0,W1,b1,W2,b2
def write_par_to_excel(W0:np.ndarray,b0:np.ndarray,W1:np.ndarray,b1:np.ndarray,W2:np.ndarray,b2:np.ndarray) -> None:
    with pd.ExcelWriter(r'Neural Network_Parameters.xlsx',mode='a',if_sheet_exists='overlay') as writer:
        pd.DataFrame(W0).to_excel(writer,sheet_name='W0',header=False,index=False) 
        pd.DataFrame(W1).to_excel(writer,sheet_name='W1',header=False,index=False)
        pd.DataFrame(W2).to_excel(writer,sheet_name='W2',header=False,index=False)
        pd.DataFrame(b0).to_excel(writer,sheet_name='b0',header=False,index=False)
        pd.DataFrame(b1).to_excel(writer,sheet_name='b1',header=False,index=False)
        pd.DataFrame(b2).to_excel(writer,sheet_name='b2',header=False,index=False)


# Activation Functions
def ReLU(mat : np.ndarray) -> np.ndarray:
    """
    Applies Rectified Linear Unit activation.
    Args:
        mat: Input matrix
    Returns:
        ReLU activated matrix
    """
    return np.maximum(mat,0)
def dff_ReLU(mat : np.ndarray) -> np.ndarray:
    return np.array(mat > 0 ,dtype=float)
def SoftMax(mat:np.ndarray) -> np.ndarray:
    """
    Applies Softmax activation along columns.
    Args:
        mat: Input matrix
    Returns:
        Probability distribution for each column
    """
    # apply softmax along columns (each column is a sample)
    return sfm(mat, axis=0)


# Forward propagation
def Run(W0:np.ndarray,b0:np.ndarray,W1:np.ndarray,b1:np.ndarray,W2:np.ndarray,b2:np.ndarray,inpu:np.ndarray) -> np.ndarray:
    inpu=inpu*(1/255) # normalize to (0,1) # inpu shape (784,m) -> m = no. of examples
    # Forward Propagation
    Z0=np.matmul(W0,inpu)+b0 #32 x m mat
    A0=ReLU(Z0) #32 x m mat
    Z1=np.matmul(W1,A0)+b1 #32 x m mat
    A1=ReLU(Z1) #32 x m mat
    Z2=np.matmul(W2,A1)+b2 #10 x m mat
    A2=SoftMax(Z2) #10 x m mat
    return A2

def one_hot_encode(labels:np.ndarray,num_classes:int=10) -> np.ndarray:
    """
    Converts integer labels to one-hot encoded format.
    Args:
        labels: Array of integer labels
        num_classes: Number of classes (default: 10)
    Returns:
        One-hot encoded matrix
    """    
    one_hot=np.zeros((num_classes,labels.shape[0]))
    for i,label in enumerate(labels):
        one_hot[label,i]=1
    return one_hot
def get_predictions(A2:np.ndarray) -> np.ndarray:
    """
    Converts output probabilities to class predictions.
    Args:
        A2: Output probability matrix
    Returns:
        Array of predicted classes
    """    
    return np.argmax(A2,0).flatten()
def get_accuracy(predictions:np.ndarray, Y:np.ndarray) -> float:
    """
    Calculates classification accuracy.
    Args:
        predictions: Model predictions
        Y: True labels
    Returns:
        Accuracy as float between 0 and 1
    """    
    # predictions: output probability matrix (num_classes, m)
    preds = get_predictions(predictions)
    labels = get_predictions(Y)
    return np.sum(preds == labels) / labels.size


class Back_propagation:
    
    def Cost_fun(self,A:np.ndarray,A_exp:np.ndarray) -> np.ndarray:
        """
        Calculates cross-entropy cost.
        Args:
            A: Predicted probabilities
            A_exp: Expected (true) values
        Returns:
            Cost per example
        """
        epsilon=1e-15
        A_clipped=np.clip(A, a_min=epsilon, a_max=1) # To Avoid log(0)
        cst=np.multiply(-A_exp,np.log(A_clipped)).mean(axis=0) # Mean over output neurons for each example
        return cst
    def Tot_cost(self,output_ : np.ndarray,Label : np.ndarray) -> float:
        totcost=self.Cost_fun(output_,Label).mean() # Mean of all examples
        return totcost 


    def Gradient_C(self,W0:np.ndarray,b0:np.ndarray,W1:np.ndarray,b1:np.ndarray,W2:np.ndarray,b2:np.ndarray,y:np.ndarray,inpu:np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        """
        Calculates gradients for all parameters.
        Args:
            Parameters: W0, b0, W1, b1, W2, b2
            y: True labels
            inpu: Input data
        Returns:
            Tuple of gradients for all parameters
        """
        m = inpu.shape[1]  # number of examples, should be 1 for single example
        inpu=(1/255.0)*inpu # normalize to (0,1) # inpu shape (784,1) or (784,m)
        Z0=np.matmul(W0,inpu)+b0 #32 x 1 mat or 32 x m mat
        A0=ReLU(Z0) #32 x 1 mat or 32 x m mat
        Z1=np.matmul(W1,A0)+b1 #32 x 1 mat or 32 x m mat
        A1=ReLU(Z1) #32 x 1 mat or 32 x m mat
        Z2=np.matmul(W2,A1)+b2 #10 x 1 mat or 10 x m mat
        A2=SoftMax(Z2) #10 x 1 mat or 10 x m mat

        # For softmax activation with cross-entropy loss, the gradient simplifies to (A2 - y)
        dCbydZ2=A2-y #10 x 1 mat or 10 x m mat
        dCbydW2=1/m*(dCbydZ2@(A1.T)) #10 x 32 mat or 10 x 32 mat and divided by m for average over m examples


        dCbydA1=(W2.T)@dCbydZ2 #32 x 1 mat or 32 x m mat
        dCbydZ1=np.multiply(dCbydA1, dff_ReLU(Z1)) #32 x 1 mat or 32 x m mat
        dCbydW1=1/m*(dCbydZ1@(A0.T)) #32 x 32 mat or 32 x 32 mat and divided by m for average over m examples


        dCbydA0=(W1.T)@dCbydZ1 #32 x 1 mat or 32 x m mat
        dCbydZ0=np.multiply(dCbydA0, dff_ReLU(Z0)) #32 x 1 mat or 32 x m mat
        dCbydW0=1/m*(dCbydZ0@(inpu.T)) #32 x 784 mat or 32 x 784 mat and divided by m for average over m examples


        # Keep bias gradients as column vectors (n,1) so broadcasting is predictable
        dCbydb2 = np.asarray(dCbydZ2).mean(axis=1).reshape(-1,1)  # 10 x 1
        dCbydb1 = np.asarray(dCbydZ1).mean(axis=1).reshape(-1,1)  # 32 x 1
        dCbydb0 = np.asarray(dCbydZ0).mean(axis=1).reshape(-1,1)  # 32 x 1

        return dCbydW0, dCbydb0, dCbydW1, dCbydb1, dCbydW2, dCbydb2
     
    # Update Parameters
    def update_par(self,W0:np.ndarray,b0:np.ndarray,W1:np.ndarray,b1:np.ndarray,W2:np.ndarray,b2:np.ndarray,delW0:np.ndarray,delb0:np.ndarray,delW1:np.ndarray,delb1:np.ndarray,delW2:np.ndarray,delb2:np.ndarray,alp:float=-1.0) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        # Update parameters in-place and return them.
        W0 += alp * delW0
        W1 += alp * delW1
        W2 += alp * delW2
        b0 += alp * delb0
        b1 += alp * delb1
        b2 += alp * delb2
        return W0, b0, W1, b1, W2, b2

def main(epochs:int,batch_size:int,al:float=0.1) -> None:
    """
    Main training function.
    Args:
        epochs: Number of training epochs
        batch_size: Number of examples per batch
        al: Learning rate
    """
    obj=Back_propagation()
    (train_inputs, train_lab), (test_in, test_lab) = mnist.load_data()
    train_inputs = train_inputs.reshape(-1,28*28).T  # Shape (784, num_train)
    test_in = test_in.reshape(-1, 28*28).T  # Shape (784, num_test)
    train_labels = one_hot_encode(train_lab, num_classes=10)  # Shape (10, num_train)
    test_lab = one_hot_encode(test_lab, num_classes=10)  # Shape (10, num_test)
    m_train = train_inputs.shape[1]
    num_batches = m_train // batch_size
    branch_to_accuracy: int = num_batches // 1 # Adjust frequency of accuracy reporting
    W0, b0, W1, b1, W2, b2 = read_par()
    for _ in range(epochs): # No of epochs
        for i in range(num_batches):
            in_ = train_inputs[:, i*batch_size:(i+1)*batch_size]
            lab_ = train_labels[:, i*batch_size:(i+1)*batch_size]
            gradients = obj.Gradient_C(W0, b0, W1, b1, W2, b2, y=np.asmatrix(lab_), inpu=np.asmatrix(in_))
            W0, b0, W1, b1, W2, b2 = obj.update_par(W0, b0, W1, b1, W2, b2, *gradients, alp=-al)
            if i % branch_to_accuracy == 0:
                print(f'{i} , Epoch ... {_}, Learning rate {al}  ')
                out = Run(W0, b0, W1, b1, W2, b2, inpu=test_in)
                print("Accuracy on Test set :", get_accuracy(out, test_lab))
    #write parameters to excel after training
    write_par_to_excel(W0,b0,W1,b1,W2,b2)
    print("Training complete")

if __name__=="__main__":
    # Uncomment the below line first time you are running this to initialize new set of random parameters
    # write_new_par()
    # Then run main training loop
    main(20,32,0.01)