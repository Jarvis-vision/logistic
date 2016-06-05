# -*- coding: utf-8 -*-
#******************#
#basic numpy 
import numpy as np
a = np.array([1, 2, 3])
print type(a)
print a.shape #3X1 vecotr

b = np.array([[1,2,3], [4,5,6]]) #2X3 matrix
print b.shape #2X3

c = np.array([[[1,2,3,4], [5,6,7,8], [9,10,11,12]],[[1,2,3,4], [5,6,7,8], [9,10,11,12]]])
print c.shape #2X3X4

print x.shape 
print np.e
#numpy: meshgrid, arrange
a = np.arange(-5, 5, 0.1)
b = np.arange(-5, 5, 0.1)
aa, bb = np.meshgrid(a, b)
print len(a), len(b)
print aa, bb

a = np.array([-1, 0, 1])
b = np.array([-2, 0, 2])
aa, bb = np.meshgrid(a, b)
aa
np.ravel(aa)
bb
np.ravel(bb)

#meshgrid and plotting 3D graph
import numpy as np
import matplotlib.pyplot as plt
points = np.arange(-5, 5, 0.01) #1,000 equally spaced vector
xs, ys = np.meshgrid(points, points)

z = np.sqrt(xs**2 + ys**2)
z

plt.imshow(z, cmap=plt.cm.gray); plt.colorbar()





#******************#
#Admission dataset******#
#******************#    
#plotting; basic EDA
from pylab import scatter, show, legend, xlabel, ylabel
#import a data
data = np.loadtxt('C:\Users\Haedong Kim\.spyder2\logistic\ex2data1.txt', delimiter=',')
print data
type(data)

x=data[:, 0:2]
y=data[:, 2]

#class labeling
pos = np.where(y==1) #return indices of records which has y==1
neg = np.where(y==0) 
print pos, neg

#plotting: x-axis: exam 1 - y-axis: exam 2
scatter(x[pos, 0], x[pos, 1], marker='o', c='b')#합격자의 시험점수 
scatter(x[neg, 0], x[neg, 1], marker='x', c='r')#불합격자의 시험점수
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend(['Not admitted', 'Admitted']) #범례
show()

#******************#
#Using scikit-learn package
#data partitioning
indices = np.random.permutation(data.shape[0]) 
training_idx, test_idx = indices[:70], indices[70:]
training_data, test_data = data[training_idx, :], data[test_idx, :]

training_x = training_data[:,0:2]
training_y = training_data[:,2]
test_x = test_data[:,0:2]
test_y = test_data[:, 2]

#learning logistic regression
from sklearn import linear_model
logit = linear_model.LogisticRegression(penalty='l2', tol=0.01, max_iter=50)
logit.fit(training_x, training_y)
print (logit.predict(test_x))

#plotting decision boundary
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

x_min, x_max = x[:, 0].min()-0.5, x[:, 0].max() + 0.5
y_min, y_max = x[:, 1].min()-0.5, x[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())                     

ax = plt.axes()
Z = logit.predict_proba(np.c_[xx.ravel(), yy.ravel(),])[:, 1]
Z = Z.reshape(xx.shape)
cs = ax.contourf(xx, yy, Z, cmap='RdBu', alpha=0.5)
cs2 = ax.contour(xx, yy, Z, cmap='RdBu', alpha=0.5)
plt.clabel(cs2, fmt = '%2.1f', colors = 'k', fontsize=14)         

ax.plot(training_x[training_y==0, 0], training_x[training_y==0, 1], 'ro', label = 'Class 1')
ax.plot(training_x[training_y==1, 0], training_x[training_y==1, 1], 'bo', label = 'Class 2') 
           
           
           
           
           
#******************#
#Iris dataset******#
#******************#
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

#import the iris dataset
iris = datasets.load_iris()
type(iris)
print iris

x = iris.data[:, 0:2] #sepal length and sepal width
y = iris.target #3 classes: Setosa, Vericolour, Virginica
#getting indices of each iris-species
setosa_idx = np.where(y==0)
vericolour_idx = np.where(y==1)
virginica_idx = np.where(y==2)

#ploting 
#plt.plot(x[:, 0], x[:, 1]) #no good
#plt.xlim(4, 8.5)
#plt.ylim(0, 5.0)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width') 
plt.title('Sepal width respect to length')                    
plt.scatter(x[setosa_idx, 0], x[setosa_idx, 1], marker='o', c='purple')
plt.scatter(x[vericolour_idx, 0], x[vericolour_idx, 1], marker='x', c='blue')
plt.scatter(x[virginica_idx, 0], x[virginica_idx, 1], marker='v', c='red')
plt.legend(['Setosa', 'Vericolour', 'Virginica'])

#partitioning dataset into a training and a test
indices = np.random.permutation(len(iris.data)) #150 records
training_idx = indices[0:105] #70% of the total number of records
test_idx = indices[105:] #30%

training_input = iris.data[training_idx, 0:2]
training_target = iris.target[training_idx]

test_input = iris.data[test_idx, 0:2]
test_target = iris.target[test_idx]

#training logistic regression classifier
logit_clf = linear_model.LogisticRegression(penalty = 'l2', max_iter = 100, solver = 'newton-cg', tol = 0.01)
logit_clf.fit(training_input, training_target) #learning done

#plotting decision boundary
x1_min, x1_max = iris.data[:, 0].min()-0.5, iris.data[:, 0].max()+0.5 #minmax of the sepal length
x2_min, x2_max = iris.data[:, 1].min()-0.5, iris.data[:, 1].max()+0.5 #minmax of the sepal width

xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.01), np.arange(x2_min, x2_max, 0.01)) #creating grid for colored plot

z = logit_clf.predict(np.c_[np.ravel(xx1), np.ravel(xx2)]) #np.ravel: python version of unlist in R / np.c_: concatenation
z = z.reshape(xx1.shape) #back to a array format

plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.pcolormesh(xx1, xx2, z, cmap=plt.cm.seismic) # choose color as change cmap option
plt.scatter(iris.data[:,0], iris.data[:,1], c=iris.target, cmap=plt.cm.seismic)

#confusion matrix
prdd = logit_clf.predict(test_input)
len(np.where(prdd==test_target))



