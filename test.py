import cvxpy as cp
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from numpy import linalg as la
from sklearn.metrics import  accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import LinearSVC
from load_datasets import *
from tensorflow.keras.utils import to_categorical

def socp(l,a,c,L,t,u,x):

    # Define the objective function for socp -> objective = cp.Minimize(t + c.T@x + l*u)
    objective = cp.Minimize((a*t) - ((1-a)*(cp.sum(cp.entr(c.T@x))) + (l*u)))

    # Sum of all optimization variable
    cs = cp.sum(cp.abs(x))

    # Define constraints
    constraint_soc = [cp.hstack([1+t,2*L@x,1-t]) >= 0, cp.sum(cs) <= u, x >= 0]

    # Problem definition
    prob = cp.Problem(objective, constraint_soc)

    # Solvers - ECOS or MOSEK
    # I would recommend to use MOSEK as solver but you can comment MOSEK part and open ECOS solver.
    prob.solve(verbose = False, solver=cp.SCS)
    # prob.solve(verbose = True, solver=cp.MOSEK)
    return x.value

def nearestPD(Q):
    """Find the nearest positive-definite matrix to input
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].
    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """
    Q = (Q + Q.T) / 2
    _, s, V = la.svd(Q)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (Q + H) / 2
    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(Q))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(Q.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def isPD(Q):
    """Returns true when input is positive-definite, via Cholesky"""

    try:
        L = la.cholesky(Q)
        return True
    except la.LinAlgError:
        return False

def create_model(number_of_features):

    model = tf.keras.models.Sequential()
    model.add(keras.layers.Dense(12, input_dim=number_of_features, activation='relu'))
    model.add(keras.layers.Dense(8, activation='relu'))
    model.add(keras.layers.Dense(5, activation='softmax'))
    # model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model

#--------------------------------------------------------------------------------------------#

if __name__ == "__main__":

    print("-------------------------------------------------")
    model = create_model(1)
    x, y = load_glioma_dataset()
    #--------------------------------------------------------------------------------------------#

    prob_train = []
    prob_test = []
    pred_train = []
    pred_test = []
    for feature in range(len(x[0])):

        # %70 = train & %30 = test
        x_train, x_test, y_train, y_test = train_test_split(x[:,feature], y, test_size=0.3, random_state=None)

        x_train = np.reshape(x_train,(len(x_train),1))
        x_test  = np.reshape(x_test, (len(x_test), 1))
        y_train = np.reshape(y_train,(len(y_train),1))
        y_test  = np.reshape(y_test, (len(y_test), 1))

        if feature == 0:
            print("\n")
            print("x_train: " + str(x_train.shape))
            print("x_test: "  + str(x_test.shape))
            print("y_train: " + str(y_train.shape))
            print("y_test:"   + str(y_test.shape))
            print("\n")

        model.fit(x_train, to_categorical(y_train), batch_size = 32, epochs = 100, verbose=0)

        probs_train_y = np.argmax(model.predict(x_train), axis=1)
        probs_test_y = np.argmax(model.predict(x_test), axis=1)
        probs_train_y = np.reshape(probs_train_y, (len(probs_train_y), 1))
        probs_test_y = np.reshape(probs_test_y, (len(probs_test_y), 1))
        print("Feature " + str(feature + 1) + ": " + "prob and pred matrix calculated")

        prob_train_array = np.array(probs_train_y)
        prob_test_array = np.array(probs_test_y)
        prob_train.append(prob_train_array)
        prob_test.append(prob_test_array)

    #--------------------------------------------------------------------------------------------#

    F_train = prob_train
    F_train = np.asarray(F_train)
    F_train = np.reshape(F_train,(F_train.shape[0],F_train.shape[1]))
    prob_train = np.asarray(prob_train)
    prob_train = np.reshape(prob_train,(prob_train.shape[0],prob_train.shape[1], prob_train.shape[2]))


    F_test = prob_test
    F_test = np.asarray(F_test)
    F_test = np.reshape(F_test,(F_test.shape[0],F_test.shape[1]))
    prob_test = np.asarray(prob_test)
    prob_test = np.reshape(prob_test,(prob_test.shape[0],prob_test.shape[1], prob_test.shape[2]))

    #--------------------------------------------------------------------------------------------#

    Q_train = np.dot(F_train, F_train.transpose())
    Q_train = nearestPD(Q_train)
    assert (isPD(Q_train))


    Q_test = np.dot(F_test, F_test.transpose())
    Q_test = nearestPD(Q_test)
    assert (isPD(Q_test))

    #--------------------------------------------------------------------------------------------#

    L_train = np.linalg.cholesky(Q_train)
    x_train = cp.Variable(len(L_train[0]))
    samples_train = prob_train.shape[1]
    models_train = prob_train.shape[0]

    L_test = np.linalg.cholesky(Q_test)
    x_test = cp.Variable(len(L_test[0]))
    samples_test = prob_test.shape[1]
    models_test = prob_test.shape[0]

    #--------------------------------------------------------------------------------------------#

    c_train = []
    for j in range(0, models_train):
        sum_samples = 0
        for i in range(0, samples_train):
            max_prob = np.max(prob_train[j][i])
            sum_samples = sum_samples + max_prob
        avg = sum_samples/F_train.shape[1]
        c_train.append(avg)
    c_train = np.asarray(c_train)
    c_train = np.reshape(c_train,(F_train.shape[0],1))


    c_test = []
    for j in range(0, models_test):
        sum_samples = 0
        for i in range(0, samples_test):
            max_prob = np.max(prob_test[j][i])
            sum_samples = sum_samples + max_prob
        avg = sum_samples/F_test.shape[1]
        c_test.append(avg)
    c_test = np.asarray(c_test)
    c_test = np.reshape(c_test,(F_test.shape[0],1))

    #--------------------------------------------------------------------------------------------#

    u = 1
    t = cp.Variable()

    l = [0.1,0.2,0.3,0.4,0.5]
    a = [0.1,0.3,0.5,0.7,0.9]

    #--------------------------------------------------------------------------------------------#


    result_train = []
    result_train_dict = dict()
    for i in range(len(l)):
        for j in range(len(a)):
            current_result = socp(l[i],a[j],c_train,L_train,t,u,x_train)
            result_train.append(current_result)
            result_train_dict[str(l[i]) + ',' + str(a[j])] = current_result
    result_train = np.asarray(result_train)
    result_train = result_train.transpose()

    result_test = []
    result_test_dict = dict()
    for i in range(len(l)):
        for j in range(len(a)):
            current_result = socp(l[i],a[j],c_test,L_test,t,u,x_test)
            result_test.append(current_result)
            result_test_dict[str(l[i]) + ',' + str(a[j])] = current_result
    result_test = np.asarray(result_test)
    result_test = result_test.transpose()

    #--------------------------------------------------------------------------------------------#

    print("\n")
    print("F_train (pred_train): " + str(F_train.shape))
    print("F_test (pred_train):  " + str(F_test.shape))
    print("prob_train:           " + str(prob_train.shape))
    print("prob_test:            " + str(prob_test.shape))
    print("Q_train:              " + str(Q_train.shape))
    print("Q_test:               " + str(Q_test.shape))
    print("c_train:              " + str(c_train.shape))
    print("c_test:               " + str(c_test.shape))
    print("result_train:         " + str(result_train.shape))
    print("result_test:          " + str(result_test.shape))
    print("experiment set:       " + str(result_train_dict.keys()))
    print("\n")

    #--------------------------------------------------------------------------------------------#

    experiment_eval = []
    weight_gain = 1
    threshold = np.mean(result_train)

    for experiment in range(result_train.shape[1]):
        weights =[]
        for model_weight in range(result_train.shape[0]):
            weights.append(result_train[model_weight][experiment]*weight_gain)

        sparsed_weights = []
        for weight in weights:
            if weight > threshold:
                sparsed_weights.append(1)
            else:
                sparsed_weights.append(0)

        sparsed_weights = np.asarray(sparsed_weights)

        new_set = []
        for k in range(x.shape[1]):
            if sparsed_weights[k] == 1:
                new_set.append(x[:,k])

        new_set = np.transpose(new_set)
        new_x = np.reshape(new_set, (x.shape[0], np.count_nonzero(sparsed_weights == 1)))

        if y.ndim == 1:
            new_y = np.reshape(y, (y.shape[0],y.ndim))
        else:
            new_y = y

        # %70 = train & %30 = test
        new_x_train, new_x_test, new_y_train, new_y_test = train_test_split(new_x, new_y, test_size=0.3, random_state=None)

        full_model = create_model(np.count_nonzero(sparsed_weights == 1))
        full_model.fit(new_x_train, to_categorical(new_y_train), batch_size = 32, epochs = 100, verbose=0)
        probs = np.argmax(full_model.predict(new_x_test), axis=1)
        probs = np.reshape(probs, (len(probs), 1))
        accuracy = accuracy_score(new_y_test, probs)
        print("Experiment " + str(experiment + 1) + ": " +
              "Remaining_features = " + str(new_x.shape[1]) + "/" + str(result_train.shape[0]) +
              " accuracy_score = " + str(accuracy))
        print('Accuracy: %.3f' % (accuracy * 100))
        experiment_eval.append(accuracy)

    best_experiment = experiment_eval.index(max(experiment_eval))
    worst_experiment = experiment_eval.index(min(experiment_eval))
    average_experiment = np.average(experiment_eval)

    print("All accuracy scores from all experiments: " + str(experiment_eval))
    print("Best experiment:         " + str(best_experiment + 1))
    print("Best experiment value:   " + str(experiment_eval[best_experiment]))
    print("Worst experiment:        " + str(worst_experiment + 1))
    print("Worst experiment:        " + str(experiment_eval[worst_experiment]))
    print("Average experiment:      " + str(average_experiment))
    print("\n")