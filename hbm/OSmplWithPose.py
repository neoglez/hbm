# -*- coding: utf-8 -*-
from __future__ import print_function
from hbm import OSmplTemplate, HumanDimensions, KJointPredictor
import numpy as np
from hbm.utils import posemap, verts_core

try:
    import cPickle as pickle
except:
    import pickle as pickle
import json

import trimesh


class OSmplWithPose:
    def __init__(
        self,
        mean_template_shape,
        shape_blendshapes,
        pose_blendshapes,
        shape=None,
        pose=None,
    ):
        assert isinstance(
            mean_template_shape, OSmplTemplate
        ), "Mean shape template must be of type OSmplTemplate"
        self.mean_template_shape = mean_template_shape
        # Betas \vec{\beta} representing linear shape coeficients.
        if shape is not None:
            assert shape.shape == (
                10,
            ), "Shape parameters do not have shape (10,)"
        else:
            shape = np.zeros(10)
        self.betas = shape
        # The matrix \mathcal{S} of shape bland shapes learned by SMPL team.
        self.shape_blendshapes = shape_blendshapes
        # The matrix \mathcal{P} of pose blend shapes learned by SMPL team.
        self.pose_blendshapes = pose_blendshapes
        # Pose here is the array of thetas representing a pose vector of
        # 72 parameters for a K = 23 joints, i.e, 3 for each part plus 3 for
        # the root orientation (3 x 23 + 3 = 72). The angles are in degrees.
        if pose is not None:
            assert pose.shape == (
                72,
            ), "Pose parameters do not have shape (72,)"
        else:
            pose = np.zeros(10)
        self.thetas = pose
        # vertices after the betas where applyed, initialized to zero
        self.vertices_shaped = np.zeros(
            (self.mean_template_shape.resolution, 3)
        )
        # vertices after the thetas where applyed
        self.vertices_posed = np.zeros(
            (self.mean_template_shape.resolution, 3)
        )
        # vertices after being reposed by the linear skinning function
        self.vertices_reposed_by_lbs = np.zeros(
            (self.mean_template_shape.resolution, 3)
        )
        # we initialize THIS body parameters as zeros.
        # This is just a simple way to hadle state: if the pose
        # or shape of this body chages (are externally setted) this variable
        # (update_body) is True. That means that you have to EXPLICITLY
        # 'update' the body in order for the skeleton to be recomputed.
        self.must_recompute_skeleton = False
        # (new in this version: joint locations after being posed)
        self.posed_joint_locations = None
        # initialize the dimensions
        self.dimensions = HumanDimensions()
        # state handling
        self.is_shaped = None
        self.is_posed = None
        self.is_lbs_skinned = None

    def update_body(self):
        # @todo handle case when the mesh DO has been posed
        # we asume linear blend skinning
        self.vertices_shaped = (
            self.mean_template_shape.vertices
            + self.shape_blendshapes.dot(self.betas)
        )

    def recompute_skeleton(self):
        # @todo handle case when the mesh DO has been posed
        joint_xyz = self.mean_template_shape.k_joint_predictor.regression_matrix.dot(self.vertices_shaped)
        return joint_xyz

    def load_from_file(self, filepath):
        filepath = ""

    def apply_shape_blend_shapes(self, betas):
        """
        Apply a set of shapes blend shapes to the model (see page 101 and
        eq 5.8) and specificaly the first 3 paragraphs on page 102.
        """
        if betas is not None:
            self.betas = betas
            self.vertices_shaped = (
                self.mean_template_shape.vertices
                + self.shape_blendshapes.dot(betas)
            )
            self.is_shaped = True
            self.must_recompute_skeleton = True

    def apply_pose_blend_shapes(self, thetas):
        """
        Apply a set of pose blend shapes to the model (see page 102 and eq 5.9)
        and specificaly the paragraphs 4-7 on page 102.
        """
        # Code adapted from the original SMPL model
        bs_type = "lrotmin"
        bs_style = "lbs"
        if thetas is not None:
            self.thetas = np.copy(thetas)
            posemap_tmp = posemap(bs_type)(thetas)
            self.vertices_posed = (
                self.vertices_shaped + self.pose_blendshapes.dot(posemap_tmp)
            )
            self.is_posed = True
            self.must_recompute_skeleton = True

    def apply_linear_blend_skinning(self):
        """
        Apply the linear blend skinnning function (see page 102 and eq 5.9)
        and specificaly the paragraphs 4-7 on page 102.
        """
        pose = self.thetas
        v = self.vertices_posed
        J = self.recompute_skeleton()
        weights = self.mean_template_shape.blend_weights
        kintree_table = self.mean_template_shape.k_intree_table
        # do we want the transformed joint locations?
        want_Jtr = True

        verts, Jtr = verts_core(pose, v, J, weights, kintree_table, want_Jtr)
        self.vertices_reposed_by_lbs = verts
        self.posed_joint_locations = Jtr
        self.is_lbs_skinned = True

    def return_template_joint_locations(self):
        """
        Return the joint locations in 3D space.
        It is a proxy to recompute_skeleton to guarantee backwards
        compatibility.

        Returns
        -------
        ndarray

        """
        return self.recompute_skeleton()

    def return_posed_template_joint_locations(self):
        """
        Return the joint locations in 3D space after the model has been posed.

        Returns
        -------
        ndarray

        """
        return self.posed_joint_locations

    def write_to_obj_file(self, outmesh_path):
        verts = []
        with open(outmesh_path, "w") as fp:
            if self.is_lbs_skinned:
                verts = self.vertices_reposed_by_lbs
            elif self.is_posed:
                verts = self.vertices_posed
            elif self.is_shaped:
                verts = self.vertices_shaped
            else:
                # just the template
                verts = self.mean_template_shape.vertices
            for v in verts:
                fp.write("v %f %f %f\n" % (v[0], v[1], v[2]))
            # write the faces containing connection/connectivity information.
            # The connectivity remains the same as the template, therefore
            # we write it to the file. Additionally, faces are 1-based, not
            # 0-based in obj files.
            for f in self.mean_template_shape.faces + 1:
                fp.write("f %d %d %d\n" % (f[0], f[1], f[2]))

    def return_as_obj_format(self):
        mesh_obj = ""
        verts = []
        if self.is_lbs_skinned:
            verts = self.vertices_reposed_by_lbs
        elif self.is_posed:
            verts = self.vertices_posed
        elif self.is_shaped:
            verts = self.vertices_shaped
        else:
            # just the template
            verts = self.mean_template_shape.vertices
        for v in verts:
            line = "v %f %f %f\n" % (v[0], v[1], v[2])
            mesh_obj = mesh_obj + line
        # write the faces containing connection/connectivity information.
        # The connectivity remains the same as the template, therefore
        # we write it to the file. Additionally, faces are 1-based, not
        # 0-based in obj files.
        for f in self.mean_template_shape.faces + 1:
            line = "f %d %d %d\n" % (f[0], f[1], f[2])
            mesh_obj = mesh_obj + line
        return mesh_obj

    def read_from_obj_file(self, inmesh_path):
        vertices = []
        for line in open(inmesh_path, "r"):
            if line.startswith("#"):
                continue
            values = line.split()
            if not values:
                continue
            if values[0] == "v":
                v = np.array([float(s) for s in values[1:4]])
                vertices.append(v)
                # we don not need to load the faces because the mesh topology
                # is always the same.
        self.vertices_shaped = np.array(vertices).reshape(
            self.mean_template_shape.resolution, 3
        )
        self.must_recompute_skeleton = True

    def write_betas_to_file(self, outbetas_path):
        with open(outbetas_path, "w") as fp:
            # Write the betas. Since "the betas" is a matrix, we have to
            # 'listifyit'.
            json.dump(
                {"betas": self.betas.tolist()},
                fp,
                sort_keys=True,
                indent=4,
                ensure_ascii=False,
            )

    def write_thetas_to_file(self, outthetas_path):
        with open(outthetas_path, "w") as fp:
            # Write the thetas. Since "the thetas" is a matrix, we have to
            # 'listifyit'.
            json.dump(
                {"thetas": self.thetas.tolist()},
                fp,
                sort_keys=True,
                indent=4,
                ensure_ascii=False,
            )

    def calculate_stature_from_mesh(self):
        # Since the mesh has LSA orientation, we continue as follows:
        # Sort the vertices by the y (Inferior-to-Superior) coordinate
        # to calculate the height, e.g., the euclidean distance between the
        # vertices with the biggest and smallest y-coordinate.
        sorted_Y = self.vertices_shaped[self.vertices_shaped[:, 1].argsort()]
        #################################################
        # Take the two vertices to compute the distance #
        #################################################
        vertex_with_smallest_y = sorted_Y[:1]
        vertex_with_biggest_y = sorted_Y[6889:]
        # Simulate the 'floor' by setting the x and z coordinate to 0.
        vertex_on_floor = np.array([0, vertex_with_smallest_y[0, 1], 0])
        stature = np.linalg.norm(
            np.subtract(
                vertex_with_biggest_y.view(dtype="f8"),
                vertex_on_floor.view(dtype="f8"),
            )
        )
        self.dimensions.stature = stature

    def write_annotations_to_file(self, annotations_path):
        with open(annotations_path, "w") as fp:
            # Write the betas. Since "the betas" is a matrix, we have to
            # 'listifyit'.
            json.dump(
                {
                    "betas": self.betas.tolist(),
                    "thetas": self.thetas.tolist(),
                    "human_dimensions": {
                        "stature": self.dimensions.stature,
                        "chest_circumference":
                            self.dimensions.chest_circumference,
                        "waist_circumference":
                            self.dimensions.waist_circumference,
                        "pelvis_circumference":
                            self.dimensions.pelvis_circumference,
                    },
                },
                fp,
                sort_keys=True,
                indent=4,
                ensure_ascii=False,
            )

    def read_annotations_from_file(self, annotations_path):
        with open(annotations_path, "r") as fp:
            data = json.load(fp)
        if len(data.get("betas", [])):
            self.betas = np.array([beta for beta in data["betas"]])
        else:
            self.betas = np.array([])
        if len(data.get("thetas", [])):
            self.thetas = np.array([theta for theta in data["thetas"]])
        else:
            self.thetas = np.array([])
        dimensions = data["human_dimensions"]
        self.dimensions = HumanDimensions(
            dimensions.get("stature"),
            dimensions.get("chest_circumference"),
            dimensions.get("waist_circumference"),
            dimensions.get("pelvis_circumference"),
        )
        # @todo handle state, e.g., the vertices_shaped has to be emptied


if __name__ == "__main__":
    # import trimesh
    # import pyglet
    SMPL_basicModel_f_lbs_path = (
        "./datageneration/smpl_data/basicModel_f_lbs_10_207_0_v1.0.0.pkl"
    )
    SMPL_basicModel_m_lbs_path = (
        "./datageneration/smpl_data/basicModel_f_lbs_10_207_0_v1.0.0.pkl"
    )

    try:
        # Load pkl created in python 2.x with python 2.x
        female_model = pickle.load(open(SMPL_basicModel_f_lbs_path, "rb"))
        male_model = pickle.load(open(SMPL_basicModel_m_lbs_path, "rb"))
    except:
        # Load pkl created in python 2.x with python 3.x
        female_model = pickle.load(
            open(SMPL_basicModel_f_lbs_path, "rb"), encoding="latin1"
        )
        male_model = pickle.load(
            open(SMPL_basicModel_m_lbs_path, "rb"), encoding="latin1"
        )

    ####################################################################
    # Initialize the joints regressor as dense array (for clarity).    #
    ####################################################################

    k_joints_predictor = female_model.get("J_regressor").A

    new_female_joint_regressor = KJointPredictor(k_joints_predictor)

    k_joints_predictor = male_model.get("J_regressor").A

    new_male_joint_regressor = KJointPredictor(k_joints_predictor)

    ####################################################################
    # Initialize the Osmpl female and male template.                   #
    ####################################################################
    new_female_template = OSmplTemplate(
        female_model.get("v_template"),
        female_model.get("f"),
        female_model.get("blend_weights"),
        female_model.get("shape_blend_shapes"),
        new_female_joint_regressor,
        female_model.get("posedirs"),
    )
    new_male_template = OSmplTemplate(
        male_model.get("v_template"),
        male_model.get("f"),
        male_model.get("blend_weights"),
        male_model.get("shape_blend_shapes"),
        new_male_joint_regressor,
        male_model.get("posedirs"),
    )

    ####################################################################
    # Once we have the template we instanciate the complete model.     #
    ####################################################################
    human_female_model = OSmplWithPose(
        new_female_template,
        female_model.get("shapedirs").x,
        female_model.get("posedirs"),
        None,
        None,
    )
    human_male_model = OSmplWithPose(
        new_male_template,
        male_model.get("shapedirs").x,
        male_model.get("posedirs"),
        None,
        None,
    )

    # attach to logger so trimesh messages will be printed to console
    # trimesh.util.attach_to_log()

    # load a file by name or from a buffer
    test_model = (
        "./CALVIS/dataset/cmu/human_body_meshes/male/male_mesh_009041.obj"
    )

    human_male_model.read_from_obj_file(test_model)
    human_male_model.calculate_all_dimensions_from_mesh()
    print(human_male_model.dimensions.stature)
    print(human_male_model.dimensions.shoulder_length)
    print(human_male_model.dimensions.hip_length)

    ########################################################################
    # Test load annotation                                                 #
    ########################################################################

    human_male_model.dimensions = None
    human_male_model.betas = None
    print("human_male_model.dimensions before loading annotation file:")
    print(human_male_model.dimensions)
    print("human_male_model.betas before loading annotation file:")
    print(human_male_model.betas)
    print("Loading annotation file now...")

    # load a file by name or from a buffer
    test_annotation_file = (
        "./CALVIS/dataset/cmu/annotations/male/male_mesh_anno_009041.json"
    )
    human_male_model.read_annotations_from_file(test_annotation_file)

    print("human_male_model.dimensions after loading annotation file:")
    print(human_male_model.dimensions.stature)
    print(human_male_model.dimensions.shoulder_length)
    print(human_male_model.dimensions.hip_length)
    print("human_male_model.betas after loading annotation file:")
    print(human_male_model.betas.shape)
    print(type(np.array(10)))

    mesh = trimesh.load(test_model)

    # is the current mesh watertight?
    print(mesh.is_watertight)

    # what's the euler number for the mesh?
    print(mesh.euler_number)

    # the convex hull is another Trimesh object that is available as a property
    # lets compare the volume of our mesh with the volume of its convex hull
    print(mesh.volume / mesh.convex_hull.volume)

    # since the mesh is watertight, it means there is a
    # volumetric center of mass which we can set as the origin for our mesh
    mesh.vertices -= mesh.center_mass

    # what's the moment of inertia for the mesh?
    mesh.moment_inertia

    # if there are multiple bodies in the mesh we can split the mesh by
    # connected components of face adjacency
    # since this example mesh is a single watertight body we get a list of one
    # mesh.split()

    # facets are groups of coplanar adjacent faces
    # set each facet to a random color
    # colors are 8 bit RGBA by default (n,4) np.uint8
    # for facet in mesh.facets:
    # mesh.visual.face_colors[facet] = trimesh.visual.random_color()

    # preview mesh in an opengl window if you installed pyglet with pip
    mesh.show()
