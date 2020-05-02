import gzip, pickle 
from gradient_descent import *
import numpy as np

with gzip.open("mnist.pkl.gz", "rb") as f:
    train_set, val_set, test_set = pickle.load(f, encoding="latin1")

X_train, y_train = train_set[0], (train_set[1] == 1).astype('int')
X_val, y_val = val_set[0], (val_set[1] == 1).astype('int')
X_test, y_test = test_set[0], (test_set[1] == 1).astype('int')

if __name__ == '__main__':

	np.random.seed(seed=42)

	B_0 = np.random.randint(-10, 10, size=X_train.shape[1]+1)
	tol_B = 10**(-6)
	tol_f = 10**(-6)
	max_iter = 10**5
	alpha = 10**(-3)


	B = grad_descent(B_0, X_train, y_train, tol_B, tol_f, max_iter, alpha)

	pred = prediction(B, X_test)

	print(pred)
	print(error(pred, y_test))


