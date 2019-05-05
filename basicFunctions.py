import numpy as np
import math

def BuildVandermondeMatrix(x,dim):
    '''1.this function is to build the Vandermonde matrix A using the vector x.
    Implementation of Vandermonde determinant based on matrix x through for loop.
    :param x:  Corresponds to the vector x in the title.
    :param dim: Number of rows in Vandermonde determinant.
    :return: Vandermonde Matrix based on matrix x.
    '''
    M = np.transpose([[(j) ** i for i in range(dim)] for j in x])
    return M

def gram_schmidt(A):
    ''' 2.this function is to use modified "Gram-schmidt" to compute a QR factorization of A.
    The core idea can be seen in Gram-schmidt method.
    :param A: Vandermonde determinant A
    :return Q: Normal orthogonal matrix Q
    :return R: Upper triangle matrix R
    '''
    Q = np.zeros_like(A)
    cnt = 0
    for a in A.T:
        u = np.copy(a)
        for i in range(0, cnt):
            u -= np.dot(np.dot(Q[:, i].T, a), Q[:, i])
        e = u / (np.sum(u ** 2, keepdims=True) ** 0.5)
        Q[:, cnt] = e
        cnt += 1
    R = np.dot(Q.T, A)
    return (Q, R)

def getinv(Q):
    '''3.this function is to compute thee inverse of Q
    This function uses the G-J elimination method to find the inverse
    :param Q: Required inverse matrix Q
    :return Q*: Inverse matrix obtained Q*
    '''
    dim = int(math.sqrt(Q.size))
    eps = 1e-6;
    mat = np.zeros([dim * 2, dim * 2])
    for i in range(0,dim):
        for j in range(0,dim*2):
            if j<dim:
                mat[i,j] = Q[i,j]
            else:
                if j-dim==i:
                    mat[i, j] = 1
                else:
                    mat[i,j] = 0

    for i in range(0,dim):
        if abs(mat[i,i]<eps):
            for j in range(i+1,dim):
                if math.fabs(mat[j,i]>eps):
                    break

            if j==dim:
                return None
            for r in range(i,2*dim):
                mat[i.r] +=mat[j,r]
        ep = mat[i,i]
        for r in range(i,2*dim):
            mat[i,r]/=ep
        for j in range(i+1,dim):
            e = -1*(mat[j,i]/mat[i,i])
            for r in range(i,2*dim):
                mat[j,r]+=e*mat[i,r]

    for i in range(dim-1,0,-1):
        for j in range(i-1,0,-1):
            e= -1*(mat[j,i]/mat[i,i])
            for r in range(i,2*dim):
                mat[j,r]+=e*mat[i,r]
    result  = np.zeros([dim,dim])
    resultt = np.linalg.inv(Q)
    for i in range(0,dim):
        for r in range(dim,2*dim):
            result[i,r-dim] = mat[i,r]
    return resultt

def SolveX(data):
    ''' 4.this function is use back substitution to solve the system
    This function mainly solves linear equations by Gaussian elimination method.
    :param data: Coefficient parameter of linear equations
    :return: Solved parameter
    '''
    i = 0;
    j = 0;
    line_size = len(data)

    while j < line_size - 1:
        line = data[j]
        temp = line[j]
        templete = []
        for x in line:
            x = x / temp
            templete.append(x)
        data[j] = templete
        flag = j + 1
        while flag < line_size:
            templete1 = []
            temp1 = data[flag][j]
            i = 0
            for x1 in data[flag]:
                if x1 != 0:
                    x1 = x1 - (temp1 * templete[i])
                    templete1.append(x1)
                else:
                    templete1.append(0)
                i += 1
            data[flag] = templete1
            flag += 1
        j += 1
    parameters = []
    i = line_size - 1
    flag_j = 0
    rol_size = len(data[0])
    flag_rol = rol_size - 2
    while i >= 0:
        operate_line = data[i]
        if i == line_size - 1:
            parameter = operate_line[rol_size - 1] / operate_line[flag_rol]
            parameters.append(parameter)
        else:
            flag_j = (rol_size - flag_rol - 2)
            temp2 = operate_line[rol_size - 1]
            result_flag = 0
            while flag_j > 0:
                temp2 -= operate_line[flag_rol + flag_j] * parameters[result_flag]
                result_flag += 1
                flag_j -= 1
            parameter = temp2 / operate_line[flag_rol]
            parameters.append(parameter)
        flag_rol -= 1
        i -= 1
    return parameters

def polynomial(coff):
    ''' this function is to construct a degree 4 interpolating polynomial and print in the screen
    :param coff:  polynomial coefficient from a
    :return: print the polynomial
    '''
    print("the degree 4 interpolating polynomial is : \n\ty={0}x^3+{1}x^2+{2}x^1+{3}".format(round(coff[0],2),round(coff[1],2),round(coff[2],2),round(coff[3],2)))
