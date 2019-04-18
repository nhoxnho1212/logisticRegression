import numpy as np
import matplotlib.pyplot as plt
import os
def sigmoid(x):
    return 1/(1+np.exp(-x))

def logistic_regression_gradient_descent(x,y,W_init,learning_Rate,tol=10e-5,max_count=10000000):
    w=W_init
    count=0
    while max_count>count:
        y_predict=sigmoid(np.dot(x.T,w))
        #GD
        gradient=np.dot(x,(y_predict-y))
        w=w-learning_Rate*gradient
        os.system('cls')
        print(str(count)+': ','%.5f'%(np.linalg.norm(gradient)),sep=' , ')
        for i in y_predict:
            print('%.3f'%(i))
        
        if np.linalg.norm(gradient)<=tol:
            return w
        
        count+=1
    
    return w


x=np.array([[0,0,1,1],[0,1,0,1]])
y=np.array([[0,0,0,1]]).T

bias=np.ones((1,x.shape[1]))

x=np.concatenate((bias,x))

w_init=np.random.randn(x.shape[0],1)

result=logistic_regression_gradient_descent(x,y,w_init,1)


print("result: ",result)
print("w_init: ",w_init)

'''
#plot
plt.plot([0,0,1],[0,1,0],'ro')
plt.plot(1,1,'bo')

def linear(w,x):
    return -(w[0][0]+w[1][0]*x)/w[2][0]
line=linear(result,np.array([0,1]))
plt.plot([0,1],line)
plt.axis([-0.5,1.5,-0.5,1.5])
plt.show()
'''