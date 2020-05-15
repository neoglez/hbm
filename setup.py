"""
Created on Tue Mar 24 14:07:43 2020

@author: neoglez
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
REQUIREMENTS = ["numpy", "opencv-python"]

setuptools.setup(
     name='hbm',  
     version='0.0.1',
     #scripts=['dokr'],
     author="Yansel Gonzalez Tejeda",
     author_email="neoglez@gmail.com",
     install_requires=REQUIREMENTS,
     description="A rapid prototyping platform focused on modeling realistic human bodies",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/neoglez/hbm",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 2",
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
     

 )

