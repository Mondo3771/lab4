import numpy as np
import matplotlib.pyplot as plt

X = np.random.normal(0 , 10 ,150) # i 
DesignMatrix = np.ones((150,3)) # ii 
for j in range(1, 3):
    for i in range(len(X)) :
        DesignMatrix[i][j] = pow(X[i],j)

#iii
Theta = np.random.uniform(0,10,3)


# iv
y_values_actual = np.matmul(DesignMatrix, Theta)
noise = np.random.normal(0,8,150)
y_values = y_values_actual+ noise
#v
plt.scatter(X , y_values_actual)
# plt.show()
# vi
X_training = DesignMatrix[:100]
y_training = y_values[:100]
X_validate = DesignMatrix[100: 125]
y_validate = y_values[100: 125]
X_Test = DesignMatrix[125:]
y_test = y_values[125]
print(X_Test)


# B)
# X_training = X[:100]
# X_validate = X[100: 125]
# X_Test = X[125:]

# Design_Matrix_training = DesignMatrix[:100]
# Desing_Matrix_validate = DesignMatrix[100: 125]
# Desing_Matrix_Test = DesignMatrix[125:]

# y_values_training = y_values[:100]
# y_values_validate = y_values[100: 125]
# y_values_Test = y_values[125:]

# Xtx = np.linalg.inv(np.matmul(Design_Matrix_training.transpose() , Design_Matrix_training))
# xty = np.matmul(Design_Matrix_training.transpose() , y_values_training)
# Theta_new = np.matmul( Xtx,xty) 
# #ii The theta's are really close to each other

# # print(Theta_new)
# # print(Theta)

# #iii
# Error = (1/2)*(np.sum((y_values_actual[:100] - y_values_training)**2))
# # print(Error)


# #iv 
# # y_values_actual = np.matmul(DesignMatrix, Theta_new)
# X_training.sort()
# Design_Matrix_training = Design_Matrix_training[Design_Matrix_training[:,1].argsort()]
# y_values_training = np.matmul(Design_Matrix_training, Theta) + noise[:100]
# y_values_actual_sorted = np.matmul(Design_Matrix_training, Theta)
# # plt.plot( X_training , y_values_training, color = "red")
# # plt.show()

# #V
alpha = 0.2

Theta_old = np.array(Theta)
Theta_old[0] += 1  
e = 0.0000000000000000000000000000000000000000000005
# Error_array =  np.array([Error])
Time_Step = np.array([0])
count=1

def descend(x,y,theta,alpha):
    dTheta = np.zeros(len(theta))
    N= len(x)
    for xi,yi in zip(x,y):
        for i in range(len(dTheta)):
            dTheta[i] = dTheta[i] -2*xi[i] * (yi - np.dot(theta,xi))  

    theta = theta - alpha *(1/N) * dTheta
    return theta

while not np.array_equal(Theta,Theta_old):
    Theta_old = np.array(Theta)
    Theta = descend(X_training,y_training,Theta,alpha) 
    count +=1
    print(count, ". Theta: ", Theta , " ThetaOld: " , Theta_old)

# plt.show()

