import numpy as np

P = np.random.rand(5,5)
P = [x / np.sum(x) for x in P] # normalization
P = np.array(P)

q = np.random.rand(5,1)
q = q / np.sum(q)
power_it = np.linalg.matrix_power(P.T, 50) # iterate the matrix first
q_50 = np.dot(power_it, q)

eig_val, eig_vec = np.linalg.eig(P.T)
stat_dist = eig_vec[:, np.isclose(eig_val, 1)] # find the index of eigenvalue 1
stat_dist = stat_dist / np.sum(stat_dist)
diff = np.linalg.norm(q_50 - stat_dist, ord=2)

if (diff < 1e-5):
    print('Match')
else:
    print('Dismatch')

