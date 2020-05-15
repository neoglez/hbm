# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 11:31:14 2018

@author: yansel
"""
from __future__ import print_function
import hbm
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle as pickle
try:
    import pymesh
except:
    pass
import json
    
class OSmpl:
    def __init__(self, mean_template_shape, shape_blendshapes, shape = [], 
                 pose = []):
        self.mean_template_shape = mean_template_shape
        # Betas \vec{\beta} representing linear shape coeficients.
        self.betas = shape
        # The matrix \mathcal{S} of shape displacements learned by SMPL team.
        self.shape_blendshapes = shape_blendshapes
        # pose here is the array of thetas representing a pose vector of 
        # 72 parameters for a K = 23 joints, i.e, 3 for each part plus 3 for 
        # the root orientation (3 x 23 + 3 = 72).
        self.thetas = pose
        # vertices after the betas where applyed
        self.vertices_shaped = []
        # vertices after the thetas where applyed
        self.vertices_posed = []
        # vertices after being reposed by the linear skinning function
        self.vertices_reposed_by_lbs = []
        # we initialize THIS body parameters as zeros.
        # This is just a simple way to hadle state: if the pose
        # or shape of this body chages (are externally setted) this variable is
        # True. That means that you have to EXPLICITLY 'update' (update_body) 
        #the body in order for the skeleton to be recomputed.
        self.must_recompute_skeleton = False
        # initialize the dimensions
        self.dimensions = hbm.HumanDimensions()
    
    def update_body(self):
        #@todo handle case when the mesh DO has been posed
        # we asume linear blend skinning
        self.vertices_shaped = (self.mean_template_shape.vertices + 
                                self.shape_blendshapes.dot(self.betas))
    
    def recompute_skeleton(self):
        #@todo handle case when the mesh DO has been posed
        joint_xyz = (
            self.mean_template_shape.k_joint_predictor.regression_matrix.dot(
                self.vertices_shaped))
        return joint_xyz
        
    def load_from_file(self, filepath):
        filepath = ''
        
    def apply_shape_blend_shapes(self, betas):
        """
        Apply a set of shapes blend shapes to the model (see page 101 and 
        eq 5.8)
        and specificaly the first 3 paragraphs on page 102.
        """
        if betas is not None:
            self.betas = betas
            self.vertices_shaped = (self.mean_template_shape.vertices + 
                                    self.shape_blendshapes.dot(betas))
            self.must_recompute_skeleton = True
    
    def apply_pose_blend_shapes(self, thetas):
        """
        Apply a set of pose blend shapes to the model (see page 102 and eq 5.9)
        and specificaly the paragraphs 4-7 on page 102.
        """
        # Code adapted from the original SMPL model
        #if posedirs is not None:
        #   v_posed = v_shaped + posedirs.dot(posemap(bs_type)(pose)) 
        if thetas is not None:
            self.thetas = thetas
            #@todo Handle the case when the model has not been shaped jet.
            self.vertices_posed = (self.vertices_shaped + 
                self.pose_blendshapes.dot(thetas))
            self.must_recompute_skeleton = True
            
    def apply_linear_blend_skinning(self):
        """
        Apply the linear blend skinnning function (see page 102 and eq 5.9)
        and specificaly the paragraphs 4-7 on page 102.
        """
        # @todo implement this
        self.vertices_reposed_by_lbs = []
        
            
    def write_to_obj_file(self, outmesh_path):
        with open( outmesh_path, 'w') as fp:
            # write (only for now) the shaped vertices
            #@todo Handle the case when the model has not been shaped jet.
            #@todo Handle the case when the model DO has been posed.
            #@todo Handle the case when the model DO has been reposed
            # by the skinning function.
            for v in self.vertices_shaped:
                fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )
            # write the faces containing connection/connectivity information.
            # The connectivity remains the same as the template, therefore
            # we write it to the file. Additionally, faces are 1-based, not 
            # 0-based in obj files.
            for f in self.mean_template_shape.faces+1:
                fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )
                
    def return_as_obj_format(self):
        # write (only for now) the shaped vertices
        #@todo Handle the case when the model has not been shaped jet.
        #@todo Handle the case when the model DO has been posed.
        #@todo Handle the case when the model DO has been reposed
        # by the skinning function.
        mesh_obj = ""
        for v in self.vertices_shaped:
            line = ( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )
            mesh_obj = mesh_obj + line
        # write the faces containing connection/connectivity information.
        # The connectivity remains the same as the template, therefore
        # we write it to the file. Additionally, faces are 1-based, not 
        # 0-based in obj files.
        for f in self.mean_template_shape.faces+1:
            line = ( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )
            mesh_obj = mesh_obj + line
        return mesh_obj
                
    def read_from_obj_file(self, inmesh_path):
        vertices = []
        for line in open(inmesh_path, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = np.array([float(s) for s in values[1:4]])
                vertices.append(v)
                # we don not need to load the faces because the mesh topology
                # is always the same.
        self.vertices_shaped = np.array(vertices).reshape(
            self.mean_template_shape.resolution, 3)
        self.must_recompute_skeleton = True
                
    def write_betas_to_file(self, outbetas_path):
        with open( outbetas_path, 'w') as fp:
            # Write the betas. Since "the betas" is a matrix, we have to 
            # 'listifyit'.
            json.dump({"betas": self.betas.tolist()}, fp, sort_keys = True, 
                      indent = 4,
                          ensure_ascii = False)

    def write_thetas_to_file(self, outthetas_path):
        with open( outthetas_path, 'w') as fp:
            json.dump({"thetas": self.thetas.tolist()}, fp, sort_keys = True, 
                      indent = 4,
                          ensure_ascii = False)
                
    def calculate_stature_from_mesh(self):
        # Since the mesh has LSA orientation, we continue as follows:       
        # Sort the vertices by the y (Inferior-to-Superior) coordinate to
        # calculate the height,
        # e.g., the euclidean distance between the vertices with the biggest
        # and smallest y-coordinate. We use the S. Tjoa's
        # (https://stackoverflow.com/users/208339/steve-tjoa) approach.
        # When comparing the height computed from the sorted array
        # by the x and the y (indexes 0 and 1) an interesting fact emerges:
        # we are able to verify the proportions of Da Vinci's Vitruvian Man :)
        # The hight is almost equal to the arms length.
        sorted_Y = self.vertices_shaped[self.vertices_shaped[:,1].argsort()]
        #################################################
        # Take the two vertices to compute the distance #
        #################################################
        vertex_with_smallest_y = sorted_Y[:1]
        vertex_with_biggest_y = sorted_Y[6889:]
        # Simulate the 'floor' by setting the x and z coordinate to 0.
        vertex_on_floor = np.array([0, vertex_with_smallest_y[0,1], 0])
        stature = np.linalg.norm(np.subtract(
            vertex_with_biggest_y.view(dtype='f8'),
                                            vertex_on_floor.view(dtype='f8')))
        self.dimensions.stature = stature
        
    def calculate_shoulder_length_from_mesh(self):
        # Default joints location is that of the template.
        # Keep in mind that if you posed or shaped the mesh
        # (must_recompute_skeleton = True), you have to reposition the joins.
        joints_location = self.mean_template_shape.joints_location
        
        if self.must_recompute_skeleton:
            joints_location = self.recompute_skeleton()
        # Since we have the position of the joints we identify the right and 
        # left shoulders and then calculate the euclidean distance 
        # between them.
        inverted_joint_names =  dict((v,k) for k,v in 
                             self.mean_template_shape.joint_names.items())
        
        shoulder_left_xyz = joints_location[inverted_joint_names['L_Shoulder']]
        shoulder_right_xyz = joints_location[
            inverted_joint_names['R_Shoulder']]
        shoulder_length = np.linalg.norm(np.subtract(
            shoulder_left_xyz.view(dtype='f8'),
            shoulder_right_xyz.view(dtype='f8')))
        self.dimensions.shoulder_length = shoulder_length
    
    def calculate_hip_length_from_mesh(self):
        # Default joints location is that of the template.
        # Keep in mind that if you posed or shaped the mesh
        # (must_recompute_skeleton = True), you have to reposition the joins.
        joints_location = self.mean_template_shape.joints_location
        
        if self.must_recompute_skeleton:
            joints_location = self.recompute_skeleton()
        
        # Since we have the position of the joints we identify the right and 
        # left shoulders and then calculate the euclidean distance 
        # between them.
        inverted_joint_names =  dict((v,k) for k,v in 
                             self.mean_template_shape.joint_names.items())
        
        hip_left_xyz = joints_location[inverted_joint_names['L_Hip']]
        hip_right_xyz = joints_location[inverted_joint_names['R_Hip']]
       
        hip_length = np.linalg.norm(np.subtract(hip_left_xyz.view(dtype='f8'),
                                            hip_right_xyz.view(dtype='f8')))
        self.dimensions.hip_length = hip_length
        
    def calculate_all_dimensions_from_mesh(self):
        # At the moment only three
        self.calculate_stature_from_mesh()
        self.calculate_shoulder_length_from_mesh()
        self.calculate_hip_length_from_mesh()
        
    
    def write_annotations_to_file(self, annotations_path):
        with open( annotations_path, 'w') as fp:
            # Write the betas. Since "the betas" is a matrix, we have to 
            # 'listifyit'.
            json.dump({
                "betas": self.betas.tolist(),
                "thetas": self.thetas.tolist(),
                "human_dimensions": {
                        "stature": self.dimensions.stature,
                        "shoulder_length": self.dimensions.shoulder_length,
                        "hip_length": self.dimensions.hip_length
                        }
                },
            fp, sort_keys = True, indent = 4, ensure_ascii = False)
    
    def read_annotations_from_file(self, annotations_path):
        with open( annotations_path, 'r') as fp:
            data = json.load(fp)
        if len(data.get('betas', [])):
            self.betas = np.array([beta for beta in data['betas']])
        else:
            self.betas = np.array([])
        if len(data.get('thetas', [])):
            self.thetas = np.array([theta for theta in data['thetas']])
        else:
            self.thetas = np.array([])
        dimensions = data['human_dimensions']
        self.dimensions = hbm.HumanDimensions(dimensions['stature'],
                                          dimensions['shoulder_length'],
                                          dimensions['hip_length'])
        #@todo handle state, e.g., the vertices_shaped has to be emptied 

if __name__ == "__main__":
    SMPL_basicModel_f_lbs_path = (
        "/media/neoglez/Data1/privat/PhD_Uni_Salzburg/"
                  "DATASETS/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl")
    SMPL_basicModel_m_lbs_path = (
        "/media/neoglez/Data1/privat/PhD_Uni_Salzburg/"
                  "DATASETS/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl")
    
    try:
        # Load pkl created in python 2.x with python 2.x
        female_model = pickle.load(open(SMPL_basicModel_f_lbs_path, 'rb'))
        male_model = pickle.load(open(SMPL_basicModel_m_lbs_path, 'rb'))
    except:
        # Load pkl created in python 2.x with python 3.x
        female_model = pickle.load(open(SMPL_basicModel_f_lbs_path, 'rb'), 
                                   encoding='latin1')
        male_model = pickle.load(open(SMPL_basicModel_m_lbs_path, 'rb'), 
                                 encoding='latin1')
        
    ####################################################################
    # Initialize the joints regressor as dense array (for clarity).    #
    ####################################################################
    
    k_joints_predictor = female_model.get('J_regressor').A
    
    new_female_joint_regressor = hbm.KJointPredictor(k_joints_predictor)
    
    k_joints_predictor = male_model.get('J_regressor').A
    
    new_male_joint_regressor = hbm.KJointPredictor(k_joints_predictor)
    
    ####################################################################
    # Initialize the Osmpl female and male template.                   #
    ####################################################################
    new_female_template = hbm.OSmplTemplate(female_model.get('v_template'),
                                        female_model.get('f'),
                                        female_model.get('blend_weights'),
                                        female_model.get('shape_blend_shapes'),
                                        new_female_joint_regressor, 
                                        female_model.get('kintree_table'),
                                        female_model.get('posedirs'))
    new_male_template = hbm.OSmplTemplate(male_model.get('v_template'),
                                        male_model.get('f'),
                                        male_model.get('blend_weights'),
                                        male_model.get('shape_blend_shapes'),
                                        new_male_joint_regressor,
                                        male_model.get('kintree_table'),
                                        male_model.get('posedirs'))
    
    ####################################################################
    # Once we have the template we instanciate the complete model.     #
    ####################################################################
    human_female_model = hbm.OSmpl(new_female_template, 
                                   female_model.get('shapedirs').x,
                               None, None)
    human_male_model = hbm.OSmpl(new_male_template, 
                                 male_model.get('shapedirs').x,
                               None, None)
    
    
    # attach to logger so trimesh messages will be printed to console
    #trimesh.util.attach_to_log()
    
    # load a file by name or from a buffer
    test_model = ('/home/neoglez/H-DIM-Project/dataset'
    '/male/male_mesh_009041.obj')
    
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
    print('human_male_model.dimensions before loading annotation file:')
    print(human_male_model.dimensions)
    print('human_male_model.betas before loading annotation file:')
    print(human_male_model.betas)
    print('Loading annotation file now...')
    
    # load a file by name or from a buffer
    test_annotation_file = (
        '/home/neoglez/H-DIM-Project/dataset/annotations/'
        'male/male_mesh_anno_009041.json')
    human_male_model.read_annotations_from_file(test_annotation_file)
    
    print('human_male_model.dimensions after loading annotation file:')
    print(human_male_model.dimensions.stature)
    print(human_male_model.dimensions.shoulder_length)
    print(human_male_model.dimensions.hip_length)
    print('human_male_model.betas after loading annotation file:')
    print(human_male_model.betas.shape)
    print(type(np.array(10)))
    
    #mesh = trimesh.load(test_model)
    
    
    # is the current mesh watertight?
    #print(mesh.is_watertight)
    
    # what's the euler number for the mesh?
    #print(mesh.euler_number)
    
    # the convex hull is another Trimesh object that is available as a property
    # lets compare the volume of our mesh with the volume of its convex hull
    #print(mesh.volume / mesh.convex_hull.volume)
    
    # since the mesh is watertight, it means there is a
    # volumetric center of mass which we can set as the origin for our mesh
    #mesh.vertices -= mesh.center_mass
    
    # what's the moment of inertia for the mesh?
    #mesh.moment_inertia
    
    # if there are multiple bodies in the mesh we can split the mesh by
    # connected components of face adjacency
    # since this example mesh is a single watertight body we get a list of one 
    # mesh
    #mesh.split()
    
    # facets are groups of coplanar adjacent faces
    # set each facet to a random color
    # colors are 8 bit RGBA by default (n,4) np.uint8
    #for facet in mesh.facets:
        #mesh.visual.face_colors[facet] = trimesh.visual.random_color()
    
    # preview mesh in an opengl window if you installed pyglet with pip
    #mesh.show()
    
    # transform method can be passed a (4,4) matrix and will cleanly apply the 
    # transform
    #mesh.apply_transform(trimesh.transformations.random_rotation_matrix())
    
    # axis aligned bounding box is available
    #mesh.bounding_box.extents
    
    # a minimum volume oriented bounding box also available
    # primitives are subclasses of Trimesh objects which automatically generate
    # faces and vertices from data stored in the 'primitive' attribute
    #mesh.bounding_box_oriented.primitive.extents
    #mesh.bounding_box_oriented.primitive.transform
    
    # show the mesh appended with its oriented bounding box
    # the bounding box is a trimesh.primitives.Box object, which subclasses
    # Trimesh and lazily evaluates to fill in vertices and faces when requested
    # (press w in viewer to see triangles)
    #(mesh + mesh.bounding_box_oriented).show()
    #mesh.show()
    
    # bounding spheres and bounding cylinders of meshes are also
    # available, and will be the minimum volume version of each
    # except in certain degenerate cases, where they will be no worse
    # than a least squares fit version of the primitive.
    #print(mesh.bounding_box_oriented.volume, 
    #      mesh.bounding_cylinder.volume,
    #      mesh.bounding_sphere.volume)