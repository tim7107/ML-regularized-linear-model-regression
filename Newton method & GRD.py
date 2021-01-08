#################################################################
#--------------------------Import-------------------------------#
#################################################################
import matplotlib.pyplot as plt
import numpy as np

#################################################################
#------------------------------DEF------------------------------#
#################################################################
def Gau_inverse(LU_inverse):
    ATA_lamdaI=np.array(LU_inverse)
    ATA_lamdaI=ATA_lamdaI.tolist()
    l=len(ATA_lamdaI)
    I=np.eye(l,dtype=int)
    #Change type from matrix to list (not array)
    I=I.tolist()
    indices=list(range(l))
    for fd in range(l):
        fdScalar= 1.0 / ATA_lamdaI[fd][fd]
        for j in range(l):
            ATA_lamdaI[fd][j] *= fdScalar
            I[fd][j] *= fdScalar
        for i in indices[0:fd] + indices[fd+1:]:
            crScalar=ATA_lamdaI[i][fd]
            for j in range(l):
                ATA_lamdaI[i][j]-=crScalar * ATA_lamdaI[fd][j]
                I[i][j]-=crScalar * I[fd][j]
    return(I)
    
#---X=LU---#
def LU_decomposition(A):
    n=len(A[0])
    L = np.zeros([n,n])
    U = np.zeros([n, n])
    for i in range(n):
        L[i][i]=1
        if i==0:
            U[0][0] = A[0][0]
            for j in range(1,n):
                U[0][j]=A[0][j]
                L[j][0]=A[j][0]/U[0][0]
        else:
                for j in range(i, n):#U
                    temp=0
                    for k in range(0, i):
                        temp = temp+L[i][k] * U[k][j]
                    U[i][j]=A[i][j]-temp
                for j in range(i+1, n):#L
                    temp = 0
                    for k in range(0, i ):
                        temp = temp + L[j][k] * U[k][i]
                    L[j][i] = (A[j][i] - temp)/U[i][i]
    return L , U

#Solve ATAX=ATb to get X=(ATA^-1)ATb
def LU_solution(L,U,b,d):    
    for i in range(d):
        b[i] = b[i]
        for j in range(d-i-1):
            k = i+1+j
            b[k] =  b[k] - b[i]*L[k][i]
    
    for i in range(d):
        h = d-i-1
        b[h] = b[h]/U[h][h]
        for j in range(d-i-1):
            k = h-1-j
            b[k] = b[k] - b[h]*U[k][h]
    return b

#traspose of matrix
def Transpose(A):
    A=np.array(A)
    AT=[[0]*n for i in range(d)]
    AT=np.array(AT)
    for i in range(len(A)):
        for j in range(len(A[0])):
            AT[j][i]=A[i][j]
    return(AT)
    
#################################################################
#--------------------------Load dataset-------------------------#
#################################################################
print("Enter lamda & d:")
lamda=int(input("lamda input= "))
d=int(input("d input= "))  #最高次方 = d-1 所以會有d項
print("\n")
data_x, data_y = np.loadtxt('C:/Users/tim/Desktop/碩一/碩一下/ML/HW01/hw01_input.txt', delimiter=',', unpack=True) #delimiter=分隔 , unpack=轉換完才能用x,y存
n=len(data_x)


##################################################################################################
#-----------------------------------------LSE method---------------------------------------------#
##################################################################################################
"""---------------------------"""
      #----Solve AX=b----#
"""---------------------------"""
#---A---#
A=[[0]*(d) for i in range(n)]
for i in range(n):
    for j in range(d):
        power = d-j-1
        A[i][j] = float(data_x[i])**power
#---b---#
b = data_y
#---x = (AtA+lamda*I)^-1 * At * b---#
"""
   temp = (AtA+lamda*I)
   LU = temp
   
"""
AT = np.transpose(A)
A    = np.transpose(AT)
I = np.identity(d)
temp = AT.dot(A)
temp = temp + I *lamda

ATb  = AT.dot(b)
L , U = LU_decomposition(temp)
#----Get X----#
X = LU_solution(L,U,ATb,d)

"""---------------------------"""
   #----print fitting line----#
"""---------------------------"""
print("----------------------------------------Result----------------------------------------")
print("LSE Fitting line :", end='')
for i in range(d):
    print(X[i],"x^",end='')
    if i == d-1:
        print(d-i-1)
    else:
        print(d-i-1,"+",end='')
        
"""---------------------------"""
        #----LSE error----#
"""---------------------------"""
predict = A.dot(X)
LSE_ERROR = 0
for i in range(n):
    LSE_ERROR = LSE_ERROR + (predict[i]-b[i])**2
print("LSE_ERROR :",LSE_ERROR)

"""---------------------------"""
        #----Plotting----#
"""---------------------------"""
x=np.linspace(-5,5)
y=np.linspace(0,100)
fitting_curve=0
for i in range(d):
    power = d-i-1
    fitting_curve=fitting_curve + X[i]*(x**power)
plt.plot(data_x, data_y, '*', label='Data Point', color='black')
plt.plot(x,fitting_curve)
plt.xlabel('x')
plt.ylabel('y')
plt.title('LSE Fitting graph')
plt.legend()
plt.show()





##################################################################################################
#----------------------------------------Newton method-------------------------------------------#
##################################################################################################
"""---------------------------"""
        #----Setting----#
"""---------------------------"""
previous = np.ones((d,1))
AT     = Transpose(A)
ATA      = AT.dot(A)
L , U    = LU_decomposition(ATA)
flag = True
for i in range(2):
    """
       ATA_previous = AT*A*X
       ATAinv_ATAX = (AT*A)^-1(AT*A*X)
    """
    ATA_previous = ATA.dot(previous)
    ATAinv_ATAX = LU_solution(L,U,ATA_previous,d)
    
    """
       ATb = AT*b
       ATAinv_ATb = (AT*A)^-1(AT*b)
    """
    ATb = AT.dot(data_y)        #ATb
    ATAinv_ATb = LU_solution(L,U,ATb,d)  #(ATA)^-1(ATb)
    ATAinv_ATb = ATAinv_ATb.reshape(d,1)
    x_new = previous - ATAinv_ATAX + ATAinv_ATb
    flag = False
    Newton_X = x_new
    Newton_X = Newton_X.reshape(1,d)
    previous = x_new


"""---------------------------"""
   #----print fitting line----#
"""---------------------------"""
print("Newton Fitting line is :", end='')
for i in range(d):
    print(Newton_X[0][i],"x^",end='')
    if i == d-1:
        print(d-i-1)
    else:
        print(d-i-1,"+",end='')

"""---------------------------"""
     #----Newton error----#
"""---------------------------"""        
Newton_X = Newton_X.reshape(d,1)
#-----------------------Newton"s method picture and error--------------------
A=np.array(A)
Newton_X=np.array(Newton_X)
predict = A.dot(Newton_X)
Newton_ERROR = 0
for i in range(n):
    Newton_ERROR = Newton_ERROR + (predict[i]-b[i])**2
print("Newton ERROR is :",Newton_ERROR[0])

"""---------------------------"""
        #----Plotting----#
"""---------------------------"""  
x=np.linspace(-5,5)
y=np.linspace(0,100)
fitting_curve=0
for i in range(d):
    power = d-i-1
    fitting_curve=fitting_curve + Newton_X[i]*(x**power)
plt.plot(data_x, data_y, '*', label='Data Point', color='black')
plt.plot(x,fitting_curve)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Newton Fitting graph')
plt.legend()
plt.show()

