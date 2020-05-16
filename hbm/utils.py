"""
Created on Tue Mar 10 09:31:04 2020

@author: neoglez
"""

import numpy as np
import cv2


def lrotmin(p):
    """
    Basically implements the substraction in equation (5.9) page 102

    Parameters
    ----------
    p : ndarray
        Pose: The pose denotes the axis-angle representation of the relative 
        rotation of part k with respect to its parent in the kinematic tree. 
        The rig has K = 23 joints, hence a pose is defined by 
        3 x 23 + 3 = 72 parameters, i.e., 3 for each part plus 3 for 
        the root orientation. Shape should be (72,)

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if isinstance(p, np.ndarray):
        # flatten and disregard the root joint
        p = p.ravel()[3:]
        # return flattened array of rotation matrices for every 3-element
        # rotation vector in pose vector
        return np.concatenate(
            [
                (
                    # rotation vector to rotation matrix through R. formula
                    cv2.Rodrigues(
                        # for every 3-elem (rotation vector)
                        np.array(pp)
                    )[0]
                    # substract the zero pose (here
                    # the identity matrix)
                    - np.eye(3)
                ).ravel()
                for pp in p.reshape((-1, 3))
            ]
        ).ravel()
    if p.ndim != 2 or p.shape[1] != 3:
        p = p.reshape((-1, 3))
    p = p[1:]
    return np.concatenate(
        [(cv2.Rodrigues(pp) - np.eye(3)).ravel() for pp in p]
    ).ravel()


def posemap(s):
    if s == "lrotmin":
        return lrotmin
    else:
        raise Exception("Unknown posemapping: %s" % (str(s),))


def global_rigid_transformation(pose, J, kintree_table):
    """
    

    Parameters
    ----------
    pose : ndarray
        Pose: The pose denotes the axis-angle representation of the relative 
        rotation of part k with respect to its parent in the kinematic tree. 
        The rig has K = 23 joints, hence a pose is defined by 
        3 x 23 + 3 = 72 parameters. Shape should be (72,) 
    J : ndarray
        Joint locations (x,y,z) in 3D space. Shape should be (24,3)
    kintree_table : TYPE
        A kinematic tree table describing the rig (tree structure ). Shape
        should be (2,24). To interpret it, it is convinient to take the
        transpose of shape (24, 2), then a row represents begin and end joints
        (index) of bones. The first row is the 'root' bone that begins in joint
        with index at infinity and ends with joint index 0. The first element
        of the row contains always the index of a parent joint.        

    Returns
    -------
    result : TYPE
        DESCRIPTION.
    results_global : TYPE
        DESCRIPTION.

    """
    results = {}
    pose = pose.reshape((-1, 3))
    # make a dict with joint indices.
    id_to_col = {kintree_table[1, i]: i for i in range(kintree_table.shape[1])}
    # make a dict with joint indices that are parents.
    parent = {
        i: id_to_col[kintree_table[0, i]]
        for i in range(1, kintree_table.shape[1])
    }

    rodrigues = lambda x: cv2.Rodrigues(x)[0]

    # lamda to construct the two buttom matrices in eq. 5.4 (page 100)
    with_zeros = lambda x: np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))
    # Construct the two upper matrices in eq. 5.4 (page 100) and assemble with
    # the bottom matrices for the first joint.
    results[0] = with_zeros(
        np.hstack((rodrigues(pose[0, :]), J[0, :].reshape((3, 1))))
    )

    # Construct upper and bottom matrices and assemble product for parents in
    # eq. 5.4 (page 100)
    for i in range(1, kintree_table.shape[1]):
        results[i] = results[parent[i]].dot(
            with_zeros(
                np.hstack(
                    (
                        rodrigues(pose[i, :]),
                        ((J[i, :] - J[parent[i], :]).reshape((3, 1))),
                    )
                )
            )
        )

    # lamda to attach a (4,1) column vector to a (4,3) matrix resulting in a
    # (4,4) matrix
    pack = lambda x: np.hstack([np.zeros((4, 3)), x.reshape((4, 1))])

    # sort
    results = [results[i] for i in sorted(results.keys())]
    results_global = results

    if True:
        results2 = [
            results[i]
            - (
                pack(
                    results[i].dot(np.concatenate(((J[i, :]), np.array([0]))))
                )
            )
            for i in range(len(results))
        ]
        results = results2
    result = np.dstack(results)
    return result, results_global


def verts_core(pose, v, J, weights, kintree_table, want_Jtr=False):
    A, A_global = global_rigid_transformation(pose, J, kintree_table)
    # apply weights
    T = A.dot(weights.T)

    rest_shape_h = np.vstack((v.T, np.ones((1, v.shape[0]))))

    v = (
        T[:, 0, :] * rest_shape_h[0, :].reshape((1, -1))
        + T[:, 1, :] * rest_shape_h[1, :].reshape((1, -1))
        + T[:, 2, :] * rest_shape_h[2, :].reshape((1, -1))
        + T[:, 3, :] * rest_shape_h[3, :].reshape((1, -1))
    ).T

    v = v[:, :3]

    if not want_Jtr:
        return v
    Jtr = np.vstack([g[:3, 3] for g in A_global])
    return (v, Jtr)
