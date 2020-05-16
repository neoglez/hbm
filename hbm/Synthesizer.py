# -*- coding: utf-8 -*-
import numpy as np

try:
    import cPickle as pickle
except:
    import pickle as pickle
try:
    import pymesh
except:
    pass
import time
import math
import hbm

from random import choice


class Synthesizer:
    def __init__(self, human_model="smpl", **kwargs):
        if human_model in ("smpl", "scape"):
            self.human_model = human_model
            self.number_of_male_models = 100000
            self.number_of_female_models = 100000
            self.smpl_female_model = None
            self.smpl_male_model = None
            self.number_of_PCAs = 10
            self.hbm_path = ""
        for key in kwargs:
            if human_model == "smpl":
                if key == "number_of_male_models":
                    self.number_of_male_models = kwargs[
                        "number_of_male_models"
                    ]
                if key == "number_of_female_models":
                    self.number_of_female_models = kwargs[
                        "number_of_female_models"
                    ]
                if key == "smpl_female_model":
                    self.smpl_female_model = kwargs["smpl_female_model"]
                if key == "smpl_male_model":
                    self.smpl_male_model = kwargs["smpl_male_model"]
                if key == "number_of_PCAs":
                    self.number_of_PCAs = kwargs["number_of_PCAs"]
                if key == "hbm_path":
                    self.hbm_path = kwargs["hbm_path"]

    def synthesize_human(self, betas, gender):
        if self.human_model == "smpl":
            # Apply the betas to the corresponding model depending on gender.
            # Note that we are synthesizing humans in the rest pose, therefore,
            # we do not have (want) to worry about the conribution of the pose
            # blend shapes (see paragraph 7 in page 102).
            if gender in ("f", "female"):
                self.smpl_female_model.apply_shape_blend_shapes(betas)
                return True
            if gender in ("m", "male"):
                self.smpl_male_model.apply_shape_blend_shapes(betas)
                return True
            raise ValueError("invalid gender: must be f, m, female or male")

    def save_human_mesh(self, gender, outmesh_path):
        if self.human_model == "smpl":
            if gender in ("f", "female"):
                self.smpl_female_model.write_to_obj_file(outmesh_path)
                return True
            if gender in ("m", "male"):
                self.smpl_male_model.write_to_obj_file(outmesh_path)
                return True
            raise ValueError("invalid gender: must be f, m, female or male")

    def return_as_obj_format(self, gender):
        if self.human_model == "smpl":
            if gender in ("f", "female"):
                return self.smpl_female_model.return_as_obj_format()
            if gender in ("m", "male"):
                return self.smpl_male_model.return_as_obj_format()
            raise ValueError("invalid gender: must be f, m, female or male")

    def save_human_dimensions(self, gender, outdimensions_path):
        if self.human_model == "smpl":
            if gender in ("f", "female"):
                self.smpl_female_model.write_to_obj_file(outdimensions_path)
                return True
            if gender in ("m", "male"):
                self.smpl_male_model.write_to_obj_file(outdimensions_path)
                return True
            raise ValueError("invalid gender: must be f, m, female or male")

    def synthesize_humans(self):
        if self.human_model == "smpl":
            # init the betas
            betas = np.zeros(self.number_of_PCAs)
            # local variables to track the synthesis
            already_synthesized_females = 0
            already_synthesized_males = 0
            genders = {0: "female", 1: "male"}
            padding_f = int(math.log10(self.number_of_female_models)) + 1
            padding_m = int(math.log10(self.number_of_male_models)) + 1

            while (
                already_synthesized_females < self.number_of_female_models
                or already_synthesized_males < self.number_of_male_models
            ):
                # now sample from a uniform distribution
                betas[:] = np.random.rand(self.number_of_PCAs)
                # pick random gender
                gender = choice(genders)
                if (
                    already_synthesized_females < self.number_of_female_models
                    and gender == "female"
                ):
                    # synthesize
                    self.synthesize_human(betas, "female")
                    # increment counter
                    already_synthesized_females += 1
                    # the name/path for this mesh
                    outmesh_path = "female_mesh_%0.*d.obj" % (
                        padding_f,
                        already_synthesized_females,
                    )
                    outmesh_path = (
                        self.hbm_path + "/dataset/female/" + outmesh_path
                    )
                    # write to file
                    self.save_human_mesh("female", outmesh_path)

                    # Calculate the height
                    self.smpl_female_model.calculate_all_dimensions_from_mesh()
                    # Write betas and height to json file
                    outjson_path = "female_mesh_anno_%0.*d.json" % (
                        padding_f,
                        already_synthesized_females,
                    )
                    outjson_path = (
                        self.hbm_path
                        + "/dataset/annotations/female/"
                        + outjson_path
                    )
                    self.smpl_female_model.write_annotations_to_file(
                        outjson_path
                    )
                    # Print message
                    print("..Annotation for mesh saved to: ", outjson_path)
                if (
                    already_synthesized_males < self.number_of_male_models
                    and gender == "male"
                ):
                    self.synthesize_human(betas, "male")
                    already_synthesized_males += 1
                    outmesh_path = "male_mesh_%0.*d.obj" % (
                        padding_m,
                        already_synthesized_males,
                    )
                    outmesh_path = (
                        self.hbm_path + "/dataset/male/" + outmesh_path
                    )
                    self.save_human_mesh("male", outmesh_path)

                    # Calculate the height
                    self.smpl_male_model.calculate_all_dimensions_from_mesh
                    # Write betas and height to json file
                    outjson_path = "./male_mesh_anno_%0.*d.json" % (
                        padding_m,
                        already_synthesized_males,
                    )
                    outjson_path = (
                        self.hbm_path
                        + "/dataset/annotations/male/"
                        + outjson_path
                    )
                    self.smpl_male_model.write_annotations_to_file(
                        outjson_path
                    )
                    # Print message
                    print("..Annotation for mesh saved to: ", outjson_path)


if __name__ == "__main__":

    ####################################################################
    # Load the models kindly provided by Loper et al., 2015.           #
    ####################################################################
    SMPL_basicModel_f_lbs_path = (
        "../datageneration/smpl_data/basicModel_f_lbs_10_207_0_v1.0.0.pkl"
    )
    SMPL_basicModel_m_lbs_path = (
        "../datageneration/smpl_data/basicmodel_m_lbs_10_207_0_v1.0.0.pkl"
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
    # Initialize the Osmpl female and male template.                   #
    ####################################################################
    new_female_template = hbm.OSmplTemplate(
        female_model.get("v_template"),
        female_model.get("f"),
        None,
        None,
        None,
        None,
    )
    new_male_template = hbm.OSmplTemplate(
        male_model.get("v_template"),
        male_model.get("f"),
        None,
        None,
        None,
        None,
    )

    ####################################################################
    # Once we have the template we instanciate the complete model.     #
    ####################################################################
    human_female_model = hbm.OSmplWithPose(
        new_female_template, female_model.get("shapedirs").x, None, None
    )
    human_male_model = hbm.OSmplWithPose(
        new_male_template, male_model.get("shapedirs").x, None, None
    )

    # Number of PCA components: The shapedirs is a tensor of shape
    # number_of_vertices x number_of_vertex_coordinate x number_of_PCA.
    # In our case this is 6890 x 3 x 10 for both female and male models.
    number_of_PCAs = female_model.get("shapedirs").shape[-1]

    hbm_dataset_path = "../"
    hbm_dataset_path = "../"

    synthesizer = Synthesizer(
        "smpl",
        number_of_male_models=1,
        number_of_female_models=1,
        smpl_female_model=human_female_model,
        smpl_male_model=human_male_model,
        number_of_PCAs=number_of_PCAs,
        hbm_path=hbm_dataset_path,
    )
    start = time.time()
    print("Starting synthesis")
    synthesizer.synthesize_humans()
    end = time.time()
    print("Finished synthesis")
    print("Time was:", end - start)
