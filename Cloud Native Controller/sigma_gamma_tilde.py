import numpy as np
import math
from scipy import linalg as la
from numpy.linalg import matrix_power


class Sigma_gamma:

    def __init__(self):
        pass

    def sigma(sigma_e):


        K=np.array([ [0.0474, 0.0321, -0.0004], [0.0321, 0.0808, -0.0114], [-0.0004, -0.0114, 0.0035] ]) #kalman gain

        P=np.array([ [0.1019e-03, 0.0735e-3, -0.0016e-03], [0.0735e-03, 0.1788e-03, -0.0251e-03], [-0.0016e-03, -0.0251e-03, 0.0072e-03]])

        R=np.array([ [0.002, 0, 0], [0, 0.002, 0], [0, 0, 0.002] ])

        A = np.array([[ 1.,0.05,-0.008756],[0.,1.,-0.3502],[0.,0.,1.]])

        B = np.array([[-6.421e-05], [-0.003853], [0.022]])

        C = np.array([[1,0.,0.], [0, 1, 0], [0., 0., 1]])

        D = np.array([[0.], [0.], [0.]])

        #LQ controller coefficient
        L=np.array([[31.0027, 13.9077, -20.4937]])  ### need to be checked

        # Q  &  R  matrices
        T_s=0.05
        sigma_w = 0.05;     # Process noise
        sigma_v = 1e-2;     # Measurement noise
        # Variances nedd to be scaled for the discrete time Kalman Filter  ?????
        var_v_d = math.pow(sigma_v,2)/T_s
        var_w_d = math.pow(sigma_w,2)*T_s
        Q = var_w_d
        R = np.eye(np.size(C,0))*var_v_d

        Q=np.eye(np.size(A,0))*var_w_d #??????


        #water marking signal
        #sigma_e=0.64
        #e=np.random.normal(0,math.sqrt(sigma_e),loop)

        #attacker's system model
        Aa_1= A+np.matmul(np.matmul(np.matmul(B,L),K),C)
        Aa_2= np.matmul(np.matmul(B,L), (np.eye(3)-np.matmul(K,C)) )
        Aa_3= np.matmul(np.matmul(B,L),K)
        Aa_4=np.matmul( (A+np.matmul(B,L)), np.matmul(K,C) )
        Aa_5=np.matmul( (A+np.matmul(B,L)), (np.eye(3)-np.matmul(K,C) ) )
        Aa_6=np.matmul( (A+np.matmul(B,L)), K )
        Aa_7=np.zeros((3,3))
        Aa_8=np.zeros((3,3))
        Aa_9=np.zeros((3,3))

        Aa=np.concatenate([np.concatenate([Aa_1,Aa_2,Aa_3],axis = 1),np.concatenate([Aa_4,Aa_5,Aa_6],axis = 1), np.concatenate([Aa_7,Aa_8,Aa_9],axis = 1)],axis = 0)

        print('Aa: ', Aa.shape)

        Ca=np.concatenate([C, np.zeros((3,3)), np.eye(3)],axis = 1)   #?????

        print('Ca: ', Ca.shape)

        Qa_1= np.matmul( np.dot(B,sigma_e),np.transpose(B) ) +Q
        Qa_2=np.matmul( np.dot(B,sigma_e),np.transpose(B) )
        Qa_3=np.zeros((3,3))
        Qa_4=np.matmul( np.dot(B,sigma_e),np.transpose(B) )
        Qa_5=np.matmul( np.dot(B,sigma_e),np.transpose(B) )
        Qa_6=np.zeros((3,3))
        Qa_7=np.zeros((3,3))
        Qa_8=np.zeros((3,3))
        Qa_9=R

        Qa=np.concatenate([np.concatenate([Qa_1,Qa_2,Qa_3],axis = 1),np.concatenate([Qa_4,Qa_5,Qa_6],axis = 1), np.concatenate([Qa_7,Qa_8,Qa_9],axis = 1)],axis = 0)

        print('Qa: ', Qa.shape)

        # Initialization of state variable under attack
        N=100
        x_a = np.zeros((9, N))
        w = np.zeros((9, N))
        z = np.zeros((3, 1))
        var_z = np.zeros((3, 3))
        sum_var_z = np.zeros((3, 3))
        var_xa = np.zeros((3,3))
        sum_var_xa = np.zeros((9, 9))
        var_xz = np.zeros((3, 3))
        mu = np.zeros((9, 1))
        mu=[0, 0, 0, 0, 0, 0, 0, 0, 0]


        for i in range(N):

            # Noise vector under attack
            w[:, i] = np.random.multivariate_normal(mu, Qa)



            # Recursion for the state variable under attack
            x_a[:, i] = np.matmul(Aa, x_a[:, i-1]) + (w[:, i])


            # Converting to column vector
            vec_xa = np.atleast_2d(x_a[:, i]).T


            # Computation of z
            z = np.matmul(Ca, vec_xa)


            # Converting to column vector
            ##vec_z = np.atleast_2d(z[:, i]).T

            # Computation of E_{zz}(0)
            ##var_z = np.matmul(vec_z, vec_z.transpose())
            var_z = np.matmul(z, z.transpose())

            sum_var_z=sum_var_z+var_z

            # Computation of E{x_a}(0)
            var_xa = np.matmul(vec_xa, vec_xa.transpose())
            sum_var_xa=sum_var_xa+var_xa

        # Scaled E_{zz}(0)
        sum_var_z /= 1/(N-1)
        print(f'var_z=', sum_var_z)
        var_zz = np.squeeze(sum_var_z)
        print(f'var_zz=', var_zz)


        # Scaled E_{x_a}(0)
        sum_var_xa /= 1/(N-1)
        print(f'var_xa=', sum_var_xa)
        var_xa=sum_var_xa

        # Computation for A hat
        A_hat =np.dot( (np.eye(3)-np.dot(K,C)),(A+np.dot(B,L)) )

        # Computation of E_{xz}(-1)
        for j in range(100000):

            var_xz_1 = np.matmul(matrix_power(A_hat, j), K)
            var_xz_2 = np.matmul(var_xz_1, Ca)
            var_xz_3 = np.matmul(var_xz_2, matrix_power(Aa, j+1))
            var_xz_4 = np.matmul(var_xz_3, var_xa)
            var_xz += np.matmul(var_xz_4, Ca.transpose())

        print(f'var_xz=', var_xz)

        # Solving Lyapunov Equation1
        Q_lya_1=np.dot( np.dot(K,var_zz),np.transpose(K) )+ np.dot( np.dot(A_hat,var_xz),np.transpose(K) )+ np.transpose( np.dot( np.dot(A_hat,var_xz), np.transpose(K) ) )

        var_xfz = la.solve_discrete_lyapunov(A_hat, Q_lya_1)
        print('var_xfz: ', var_xfz)

        # Solving Lyapunov Equation2
        term1=np.eye(3)-np.dot(K,C)
        term2= np.dot( np.dot(np.dot(term1,B),sigma_e), np.transpose(B) )
        Q_lya_2=np.dot(term2, np.transpose(term1))

        var_xfe = la.solve_discrete_lyapunov(A_hat, Q_lya_2)
        print('var_xfe: ', var_xfe)


        part1=var_zz
        part2=np.dot(np.dot( C,(A+np.dot(B,L)) ), var_xz)
        part3=np.transpose(part2)
        part4=np.dot( np.dot( np.dot(np.dot(C,B),sigma_e),np.transpose(B) ), np.transpose(C) )

        part5_a=A+np.dot(B,L)
        part5=np.dot( np.dot( np.dot(np.dot(C,part5_a), var_xfz ),np.transpose(part5_a) ), np.transpose(C) )

        part6=np.dot( np.dot( np.dot(np.dot(C,part5_a), var_xfe ),np.transpose(part5_a) ), np.transpose(C) )

        sigma_gamma_tilde=part1+part2+part3+part4+part5+part6

        print('sigma_gamma_tilde: ', sigma_gamma_tilde)

        return sigma_gamma_tilde

