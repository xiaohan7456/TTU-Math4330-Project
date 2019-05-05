import numpy as np
import basicFunctions

if __name__ == "__main__":
    # input vector x , y .if you want to create degree 4 interpolating polynomial ,size_x , size_y must be (4.) e.g.
    x = np.array([1,2,3,4],dtype=np.float)
    y = np.array([8,7,6,5],dtype=np.float)
    dim = x.size  # get size_x and size_y to cofficient dim
    # <1> throught basicFunction.BuildVandermodeMatrix to create x VandermondeMatrix
    M = basicFunctions.BuildVandermondeMatrix(x,dim)
    print(M)
    # <2> throught basicFunctions.gram_schmidt to get QR matrix from M
    (Q,R) = basicFunctions.gram_schmidt(M)
    # <3> throught basicFunctions.getinv to get inverse Matrix of Q
    invQ = basicFunctions.getinv(Q)
    print(invQ)
    temp1 = np.dot(invQ,y)
    coff = np.c_[R,temp1]
    # <4> throught basicFunctions.SolveX to get system root.
    a = basicFunctions.SolveX(coff)
    # <5> throught basicFunctions.polynomial to print degree 4 interpolating polynomial
    basicFunctions.polynomial(a)