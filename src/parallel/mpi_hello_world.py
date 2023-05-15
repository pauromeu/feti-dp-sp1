from mpi4py import MPI
import numpy as np


def naive_mat_vec_mult(matrix, vector):
    result = np.zeros((matrix.shape[0], 1))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            result[i] += matrix[i, j] * vector[j]
    return result


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 1000
A = np.random.rand(N, N)
x = np.random.rand(N, 1)

start = MPI.Wtime()

chunk = N // size
low = rank * chunk
high = (rank + 1) * chunk if rank != size - 1 else N

result_local = naive_mat_vec_mult(A[low:high, :], x)

result = None
if rank == 0:
    result = np.empty_like(result_local)

comm.Reduce(result_local, result, op=MPI.SUM, root=0)

end = MPI.Wtime()
elapsed_time = end - start

if rank == 0:
    print(f"Elapsed time with {size} processes: {elapsed_time} seconds")
