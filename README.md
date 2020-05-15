### About HBM ###

**Human Body Models (HBM)** is a code base developed by Yansel González Tejeda for his PhD research at Paris Lodron University of Salzburg (graduation expected on winter 2020).  It is a rapid prototyping platform focused on modeling realistic human bodies. It contains generative models based on the successfully SCAPE and SMPL models.  **HBM** is written with Python in a tremendously "chatty" way to make it especially easy to understand to people with all academic backgrounds (including those without it). Therefore, this code is not recommended for production purposes, you can use it at your own risk though.

An appropiate starting point are the following tutorials:

**SCAPE Model (Shape Completion and Animation of PEople)**
* Visualize a human SCAPE model (visualize the mesh and the skeleton).
* Visualize human SCAPE model rigid parts (rigid parts are surface mesh points whose shape is preserved over time, i.e., an arm or a leg).
* Visualize adjacent parts angles (visualize the angle between the right arm and the right shoulder).
* Synthesize a new 3D body mesh.
* Synthesize a 3D mesh in a new pose (standing with the left arm strechted to the side and the right arm resting beside the body).
* Synthesize a 3D mesh with a new body in a new pose (full pipeline).
* The inverse problem: fit the SCAPE model to a human 3D mesh scan.

**SMPL Model (Skinned Multiperson Linear)**
* Visualize a human SMPL model (visualize the mesh and the skeleton).
* Visualize human SMPL model rigid parts (rigid parts are surface mesh points whose shape is preserved over time, i.e., an arm or a leg).
* Visualize adjacent parts angles (visualize the angle between the right arm and the right shoulder).
* Synthesize a new 3D body mesh in the normal pose.
* Visualize the newly synthesized mesh with trimesh.
* Synthesize a 3D mesh in a new pose (standing with the left arm strechted to the side and the right arm resting beside the body).
* Synthesize a 3D mesh with a new body in a new pose (full pipeline).
* The inverse problem: fit the SMPL model to a human 3D mesh scan. 

## Acknowledgements ##
The SMPL team for the trainned components of the model.
The SCAPE team for the trainned components of the model.
The data generation code is inspired by [Gul Varol's SURREAL](https://github.com/gulvarol/surreal).
The data loading code is inspired by [3D ResNets for Action Recognition](https://github.com/kenshohara/3D-ResNets-PyTorch).
