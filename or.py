import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def Or_logisticRegression_GradientDescent(x,y,w_init,learning_rate,tol=10e-4,max_count=10e4):
    max_count=int(max_count)
    w=w_init
    
    for i in range(max_count):
        y_predict=sigmoid(np.dot(x.T,w))
        gradient=np.dot(x,y_predict-y)
        w=w-learning_rate*gradient
        if np.linalg.norm(gradient)<tol:
            return w
        print(i,' :',end=' ')
        print(w,np.linalg.norm(gradient),sep=' , ')
    return w        


x=np.array([[0,0,1,1],[0,1,0,1]])
y=np.array([[0,1,1,1]]).T

bias=np.ones((1,x.shape[1]))
x=np.concatenate((bias,x))

w_init=np.random.randn(x.shape[0],1)
learning_rate=0.5
result=Or_logisticRegression_GradientDescent(x,y,w_init,learning_rate)

print('\nsuccess!!\n')

for i in result:
    print(i)