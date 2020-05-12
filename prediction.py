import gzip, pickle 
from gradient_descent import *
import numpy as np

with gzip.open("mnist.pkl.gz", "rb") as f:
    train_set, val_set, test_set = pickle.load(f, encoding="latin1")

# Only take 0's and 1's
indx_trn = np.where((train_set[1] == 0) | (train_set[1] == 1))
indx_vld = np.where((val_set[1] == 0) | (val_set[1] == 1))
indx_tst = np.where((test_set[1] == 0) | (test_set[1] == 1))

train_set = (train_set[0][indx_trn], train_set[1][indx_trn])
val_set = (val_set[0][indx_vld], val_set[1][indx_vld])
test_set = (test_set[0][indx_tst], test_set[1][indx_tst])

print('Train Shape : ', train_set[0].shape, train_set[1].shape)
print('Valid Shape : ', val_set[0].shape, val_set[1].shape)
print('Test Shape : ', test_set[0].shape, test_set[1].shape)

X_train, y_train = train_set[0], (train_set[1] == 1).astype('int')
X_val, y_val = val_set[0], (val_set[1] == 1).astype('int')
X_test, y_test = test_set[0], (test_set[1] == 1).astype('int')

if __name__ == '__main__':

	np.random.seed(seed=42)

	B_0 = np.random.randint(-10, 10, size=X_train.shape[1]+1)
	tol_B = 10**(-3)
	tol_f = 10**(-3)
	max_iter = 10**5
	alpha = 10**(-3)


	B = grad_descent(B_0, X_train, y_train, tol_B, tol_f, max_iter, alpha)

	pred = prediction(B, X_test)

	print(pred)
	print('err', error(pred, y_test))
	print('acc', 1.0 - error(pred, y_test))
	print(confusion_matrix(y_test, pred))

