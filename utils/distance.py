"""Numpy version of euclidean distance, shortest distance, etc.
Notice the input/output shape of methods, so that you can better understand
the meaning of these methods."""
import numpy as np
import torch
import time
import sys

def normalize(nparray, order=2, axis=0):
    """Normalize a N-D numpy array along the specified axis."""
    norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
    return nparray / (norm + np.finfo(np.float32).eps)


def compute_dist(array1, array2, type='euclidean'):
    """Compute the euclidean or cosine distance of all pairs.
  Args:
    array1: numpy array with shape [m1, n]
    array2: numpy array with shape [m2, n]
    type: one of ['cosine', 'euclidean']
  Returns:
    numpy array with shape [m1, m2]
  """
    assert type in ['cosine', 'euclidean']
    if type == 'cosine':
        array1 = normalize(array1, axis=1)
        array2 = normalize(array2, axis=1)
        dist = np.matmul(array1, array2.T)
        return dist
    else:
        # shape [m1, 1]
        square1 = np.sum(np.square(array1), axis=1)[..., np.newaxis]
        # shape [1, m2]
        square2 = np.sum(np.square(array2), axis=1)[np.newaxis, ...]
        squared_dist = - 2 * np.matmul(array1, array2.T) + square1 + square2
        squared_dist[squared_dist < 0] = 0
        dist = np.sqrt(squared_dist)
        return dist


def shortest_dist(dist_mat):
  """Parallel version.
  Args:
    dist_mat: pytorch Variable, available shape:
      1) [m, n]
      2) [m, n, N], N is batch size
      3) [m, n, *], * can be arbitrary additional dimensions
  Returns:
    dist: three cases corresponding to `dist_mat`:
      1) scalar
      2) pytorch Variable, with shape [N]
      3) pytorch Variable, with shape [*]
  """
  m, n = dist_mat.size()[:2]
  # Just offering some reference for accessing intermediate distance.
  dist = [[0 for _ in range(n)] for _ in range(m)]
  for i in range(m):
    for j in range(n):
      if (i == 0) and (j == 0):
        dist[i][j] = dist_mat[i, j]
      elif (i == 0) and (j > 0):
        dist[i][j] = dist[i][j - 1] + dist_mat[i, j]
      elif (i > 0) and (j == 0):
        dist[i][j] = dist[i - 1][j] + dist_mat[i, j]
      else:
        dist[i][j] = torch.min(dist[i - 1][j], dist[i][j - 1]) + dist_mat[i, j]
  dist = dist[-1][-1]
  return dist

def local_dist(x, y, aligned):
    """Parallel version.
  Args:
    x: numpy array, with shape [M, m, d]
    y: numpy array, with shape [N, n, d]
  Returns:
    dist: numpy array, with shape [M, N]
  """
    M, m, d = x.size()
    N, n, d = y.size()

    #x = torch.tensor(x).cuda()
    #y = torch.tensor(y).cuda()

    x = x.view(M*m, d)
    y = y.view(N*n, d)

    #x = x.reshape([M * m, d])
    #y = y.reshape([N * n, d])
    # shape [M * m, N * n]
    #dist_mat = compute_dist(x, y, type='euclidean')

    dist_mat = torch.cdist(x, y)

    #dist_mat = (np.exp(dist_mat) - 1.) / (np.exp(dist_mat) + 1.)

    dist_mat = (torch.exp(dist_mat) - 1.) / (torch.exp(dist_mat) + 1.)

    # shape [M * m, N * n] -> [M, m, N, n] -> [m, n, M, N]
    dist_mat = dist_mat.reshape([M, m, N, n]).permute([1, 3, 0, 2])
    # shape [M, N]
    if aligned:
        dist_mat = shortest_dist(dist_mat)
    else:
        raise Exception("Only aligned dist is implemented!")
    return dist_mat

def low_memory_matrix_op(
        func,
        x, y,
        x_split_axis, y_split_axis,
        x_num_splits, y_num_splits,
        verbose=False, aligned=True):
    """
  For matrix operation like multiplication, in order not to flood the memory
  with huge data, split matrices into smaller parts (Divide and Conquer).

  Note:
    If still out of memory, increase `*_num_splits`.

  Args:
    func: a matrix function func(x, y) -> z with shape [M, N]
    x: numpy array, the dimension to split has length M
    y: numpy array, the dimension to split has length N
    x_split_axis: The axis to split x into parts
    y_split_axis: The axis to split y into parts
    x_num_splits: number of splits. 1 <= x_num_splits <= M
    y_num_splits: number of splits. 1 <= y_num_splits <= N
    verbose: whether to print the progress

  Returns:
    mat: numpy array, shape [M, N]
  """

    if verbose:
        printed = False
        st = time.time()
        last_time = time.time()

    mat = [[] for _ in range(x_num_splits)]
    for i, part_x in enumerate(
            np.array_split(x, x_num_splits, axis=x_split_axis)):
        for j, part_y in enumerate(
                np.array_split(y, y_num_splits, axis=y_split_axis)):
            part_mat = func(part_x, part_y, aligned)
            mat[i].append(part_mat)

            if verbose:
                if not printed:
                    printed = True
                else:
                    # Clean the current line
                    sys.stdout.write("\033[F\033[K")
                print('Matrix part ({}, {}) / ({}, {}), +{:.2f}s, total {:.2f}s'
                    .format(i + 1, j + 1, x_num_splits, y_num_splits,
                            time.time() - last_time, time.time() - st))
                last_time = time.time()
        mat[i] = np.concatenate(mat[i], axis=1)
    mat = np.concatenate(mat, axis=0)
    return mat


def low_memory_local_dist(x, y, aligned=True):
    print('Computing local distance...')

    slice_size = 200 # old 200
    x_num_splits = int(len(x) / slice_size) + 1
    y_num_splits = int(len(y) / slice_size) + 1
    z = low_memory_matrix_op(local_dist, x, y, 0, 0, x_num_splits, y_num_splits, verbose=True, aligned=aligned)
    return z



def compute_local_distmat_using_gpu(probFea, galFea, memory_save=True, mini_batch=2000):
    print('Computing distance using GPU ...')
    feat = torch.cat([probFea, galFea]).cuda()
    all_num = probFea.size(0) + galFea.size(0)
    if memory_save:
        distmat = torch.zeros((all_num, all_num), dtype=torch.float16)  # 14 GB memory on Round2
        i = 0
        while True:
            it = i + mini_batch
            # print('i, it', i, it)
            if it < feat.size()[0]:
                distmat[i:it, :] = torch.pow(torch.cdist(feat[i:it, :], feat), 2)
            else:
                distmat[i:, :] = torch.pow(torch.cdist(feat[i:, :], feat), 2)
                break
            i = it
    else:
        ### new API
        distmat = torch.pow(torch.cdist(feat, feat), 2)

    # print('Copy distmat to original_dist ...')
    original_dist = distmat.numpy()  # 14 GB memory
    del distmat
    del feat
    return original_dist


def fast_local_dist(x, y, aligned=True, verbose=True):
    print('Computing local distance...')

    slice_size = 200 # old 200

    if verbose:
        #import sys

        printed = False
        st = time.time()
        last_time = time.time()

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    feat = torch.cat([x, y]).cuda()
    all_num = feat.size(0)

    mat = np.zeros((all_num, all_num), dtype=np.float16)

    x_num_splits = int(all_num / slice_size) + 1
    y_num_splits = int(all_num / slice_size) + 1

    pi = 0
    for i, part_x in enumerate(torch.split(feat, slice_size, dim=0)):
        # move to cuda
        dx = part_x.size(0)
        pj = 0
        for j, part_y in enumerate(torch.split(feat, slice_size, dim=0)):
            # move to cuda
            dy = part_y.size(0)
            #print('dx, dy', dx, dy)
            # compute using gpu
            #print('part_x', part_x.size(), 'part_y', part_y.size())
            part_mat = local_dist(part_x, part_y, aligned)
            #print('part_mat', part_mat.size())
            # fetch result
            part_mat = part_mat.cpu().numpy()

            mat[pi:pi+dx, pj:pj+dy] = part_mat
            pj += dy

            if verbose:
                if not printed:
                    printed = True
                else:
                    # Clean the current line
                    sys.stdout.write("\033[F\033[K")
                print('Matrix part ({}, {}) / ({}, {}), +{:.2f}s, total {:.2f}s'
                    .format(i + 1, j + 1, x_num_splits, y_num_splits,
                            time.time() - last_time, time.time() - st))
                last_time = time.time()

        pi += dx

    return mat




