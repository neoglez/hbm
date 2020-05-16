"""
Created on Tue Mar 24 15:00:37 2020

@author: neoglez
"""


class OSmplTemplate:
    def __init__(
        self,
        vertices,
        faces,
        blend_weights,
        blend_shape_function,
        k_joint_predictor,
        k_intree_table,
        pose_dependant_blend_shape_function,
    ):
        self.vertices = vertices
        # resolution is the number of vertices
        self.resolution = len(self.vertices)
        self.faces = faces
        self.blend_weights = blend_weights
        self.blend_shape_function = blend_shape_function
        self.k_joint_predictor = k_joint_predictor
        self.k_intree_table = k_intree_table
        self.pose_dependant_blend_shape_function = (
            pose_dependant_blend_shape_function
        )
        # Code adapted from SMPL maya plugin version 1.0.2
        # Dictionary for matching joint index to name
        self.joint_names = {
            0: "Pelvis",
            1: "L_Hip",
            4: "L_Knee",
            7: "L_Ankle",
            10: "L_Foot",
            2: "R_Hip",
            5: "R_Knee",
            8: "R_Ankle",
            11: "R_Foot",
            3: "Spine1",
            6: "Spine2",
            9: "Spine3",
            12: "Neck",
            15: "Head",
            13: "L_Collar",
            16: "L_Shoulder",
            18: "L_Elbow",
            20: "L_Wrist",
            22: "L_Hand",
            14: "R_Collar",
            17: "R_Shoulder",
            19: "R_Elbow",
            21: "R_Wrist",
            23: "R_Hand",
        }
        # define explicitly the join locations
        self.joints_location = self.compute_joints_location()

    def load_from_file(self, filepath):
        filepath = ""

    def compute_joints_location(self):
        # regress joints position in rest pose
        joint_xyz = self.k_joint_predictor.regression_matrix.dot(self.vertices)
        return joint_xyz
