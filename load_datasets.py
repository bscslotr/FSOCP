from enum import unique
import scipy.io
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize
import numpy as np

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# -------------------- Yale.mat data --------------------

def load_yale_dataset():
    mat_data = scipy.io.loadmat('Yale.mat')
    x = mat_data['X']
    x = x.astype(float)
    # normalized_x = normalize(x)
    normalized_x = NormalizeData(x)
    y = mat_data['Y']
    y = y[:, 0]
    y = np.reshape(y,(len(y),1))
    print("x -> (Samples, Features) -> " + str(x.shape))
    print("y -> (Samples,) -> " + str(y.shape))
    print("Number of classes -> " + str(np.unique(y)))
    return normalized_x, y

# -------------------- Isolet.mat data --------------------

def load_isolet_dataset():
    mat_data = scipy.io.loadmat('Isolet.mat')
    x = mat_data['X']
    x = x.astype(float)
    y = mat_data['Y']
    y = y[:, 0]
    print("x -> (Samples, Features) -> " + str(x.shape))
    print("y -> (Samples,) -> " + str(y.shape))
    print("Number of classes -> " + str(np.unique(y)))
    return x, y

# -------------------- lung_small.mat data --------------------

def load_lung_small_dataset():
    # mat_data = scipy.io.loadmat('lung_small.mat')
    mat_data = scipy.io.loadmat("lung_small.mat")
    x = mat_data['X']
    x = x.astype(float)
    # normalized_x = normalize(x)
    normalized_x = NormalizeData(x)
    y = mat_data['Y']
    y = y[:, 0]
    y = np.reshape(y,(len(y),1))
    print("x -> (Samples, Features) -> " + str(x.shape))
    print("y -> (Samples,) -> " + str(y.shape))
    print("Number of classes -> " + str(np.unique(y)))
    return normalized_x, y

# -------------------- colon.mat data --------------------

def load_colon_dataset():
    mat_data = scipy.io.loadmat('colon.mat')
    x = mat_data['X']
    x = x.astype(float)
    normalized_x = NormalizeData(x)
    y = mat_data['Y']
    y = y[:, 0]
    print("x -> (Samples, Features) -> " + str(x.shape))
    print("y -> (Samples,) -> " + str(y.shape))
    print("Number of classes -> " + str(np.unique(y)))
    return normalized_x, y

# -------------------- madelon.mat data --------------------

def load_glioma_dataset():
    mat_data = scipy.io.loadmat('GLIOMA.mat')
    x = mat_data['X']
    x = x.astype(float)
    normalized_x = NormalizeData(x)
    y = mat_data['Y']
    y = y[:, 0]
    print("x -> (Samples, Features) -> " + str(x.shape))
    print("y -> (Samples,) -> " + str(y.shape))
    print("Number of classes -> " + str(np.unique(y)))
    return normalized_x, y

# -------------------- breast cancer data --------------------

def load_breast_cancer_dataset():
    data_file = "data.csv"
    df = pd.read_csv(data_file)

    print(df.isnull().sum())
    df.drop(['Unnamed: 32','id'],axis=1,inplace=True)

    x = df.drop('diagnosis',axis=1)
    y = df.diagnosis

    lb = LabelEncoder()
    y = lb.fit_transform(y)

    x = x.to_numpy()
    print("x -> (Samples, Features) -> " + str(x.shape))
    print("y -> (Samples,) -> " + str(y.shape))
    print("Number of classes -> " + str(np.unique(y)))
    return x, y

# -------------------- diabetes data --------------------

def load_diabetes_dataset():
    data_file = "diabetes.csv"
    df = pd.read_csv(data_file)
    x = df.to_numpy()[:, 0:8]
    y = df.to_numpy()[:, 8]
    print("x -> (Samples, Features) -> " + str(x.shape))
    print("y -> (Samples,) -> " + str(y.shape))
    print("Number of classes -> " + str(np.unique(y)))
    return x, y

# -------------------- ionosphere data --------------------

def load_ionosphere_dataset():
    data_file = "ionosphere.csv"
    df = pd.read_csv(data_file)

    def rename(x):
        if 'feature' in x:
            return x.replace("feature", "f")
        return x

    target_map = {"g" : 1, "b" : 0}
    df["label"] = df.label.map(target_map).astype(np.int)
    df = df.rename(columns = rename)
    x = df.drop("label", axis = 1)
    y = df.label
    x = x.to_numpy()
    y = y.to_numpy()
    print("x -> (Samples, Features) -> " + str(x.shape))
    print("y -> (Samples,) -> " + str(y.shape))
    print("Number of classes -> " + str(np.unique(y)))
    return x, y

#--------------------------------------------------------------------------------------------#