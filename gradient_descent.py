import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


#
def pi(B, X):



	res = []
	#print(X.shape, B.shape)
	for i in range(X.shape[0]):
		#print(X[i].shape)
		res.append(1/(1 + np.exp(-X[i].T@B)))


	return np.array(res)

# We want to minimiimport gzip, pickle ze this function
def func(B, pi, X, y):

	n = X.shape[0]
	pi_1 = pi.copy()

	func = 0

	for i in range(n):

		eps = 1e-5
		if np.isclose(pi_1[i], 0.0):
			pi_1[i] = eps
		
		func+= y[i]*np.log(pi_1[i])

		if np.isclose(pi_1[i], 1.0):
			pi_1[i] = 1-eps
		
		func+= (1-y[i])*np.log(1-pi_1[i])

	return -func

def grad(B, pi, X, y):

	n = X.shape[0]
	m = X.shape[1]


	#Extra feature for bias
	grad = np.zeros(m, dtype=float)

	for i in range(n):

		grad += (y[i] - pi[i]) * X[i,:]

	return grad


def grad_descent(B_0, X, y, tol_B, tol_f, max_iter, alpha):
    
    k = 0

    n = X.shape[0]
    #Add extra dimension for bias
    ones = np.ones((n, 1))
    X = np.concatenate((X,ones), axis=1)
    dis_f = float('inf')
    dis_B = float('inf')
    B = B_0

    pi_ = pi(B, X)

    
    while k < max_iter and dis_f >= tol_f and dis_B >= tol_B:
        
        d = grad(B, pi_, X, y)

        #d /= np.linalg.norm(d)

        B_new = B - alpha*d

        
        print('d:{}'.format(np.linalg.norm(d)))
        print('B: {}'.format(np.linalg.norm(B)))
        print('B_new: {}'.format(np.linalg.norm(B_new)))
        

        pi_new = pi(B_new, X)
    
        f_x = func(B, pi_, X, y)
        f_new = func(B_new, pi_new, X, y)

        
        print('f_x: {}'.format(f_x))
       	print('f_new: {}'.format(f_new))
       	

        
        #if f_new < f_x:

        print('\nIter: {}'.format(k))

        B_old = B

        B = B_new

        pi_ = pi(B, X)

        k += 1

        dis_B = np.linalg.norm(B - B_old)/max(1, np.linalg.norm(B_old))

        dis_f = np.abs(f_new - f_x)/max(1,np.abs(f_x))

    
    return B 

def prediction(B, X):

	n = X.shape[0]
	ones = np.ones((n, 1))
	X = np.concatenate((X,ones), axis=1)

	pred =  X@B
	#Change the value w/ value above 0.5 to 1 and 0 otherwise
	pred = np.where(pred > 0.5, 1, 0)
	return pred

def error(pred, y):

	n = len(y)

	err = 0
	for i in range(n):
		err += np.abs(pred[i] - y[i])

	return err/n 



if __name__== '__main__':


	np.random.seed(seed=42)

	
	X, y = make_classification(n_features=5)


	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	
	
	B_0 = np.random.randint(-10, 10, size=X_train.shape[1]+1)
	tol_B = 10**(-6)
	tol_f = 10**(-6)
	max_iter = 10**5
	alpha = 10**(-3)


	B = grad_descent(B_0, X_train, y_train, tol_B, tol_f, max_iter, alpha)

	pred = prediction(B, X_test)

	print(pred)
	print(error(pred, y_test))






