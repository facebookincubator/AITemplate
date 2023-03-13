#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import os

def mark_output(y):
    if type(y) is not tuple:
        y = (y,)
    for i in range(len(y)):
        y[i]._attrs["is_output"] = True
        y[i]._attrs["name"] = "output_%d" % (i)
        y_shape = [d._attrs["values"][0] for d in y[i]._attrs["shape"]]
        print("AIT output_{} shape: {}".format(i, y_shape))
        
def get_work_dir_location():
    
    """
    Set the OS environment variable AITEMPLATE_WORK_DIR to point to an absolute
    path to a directory which has AITemplate compiled artifacts the model(s). 
    Make sure the OS user running this script has read and write permissions to 
    this directory. By default, the artifacts will be saved under tmp/ folder of 
    the current working directory. 
    """
        
    env_name = "AITEMPLATE_WORK_DIR"
    workdir = "tmp/"    
    if env_name in os.environ:
        workdir = os.environ[env_name]     
            
    print("The value of {} is {}".format(env_name, workdir))  
    
    return workdir

def get_work_dir_location_diffusers():
    
    """
    Set the OS environment variable AITEMPLATE_WORK_DIR to point to an absolute
    path to a directory which has AITemplate compiled artifacts the model(s). 
    Make sure the OS user running this script has read and write permissions to 
    this directory. By default, the it will look for compiled artifacts under 
    tmp/ folder of the current working directory. 
    
    """
    
    env_name = "AITEMPLATE_WORK_DIR"
    local_dir = "./tmp/diffusers-pipeline/stabilityai/stable-diffusion-v2"
    
    if env_name in os.environ:
        local_dir = os.path.join(os.environ[env_name], 'diffusers-pipeline', 'stabilityai', 'stable-diffusion-v2')
          
    print("The value of {} is {}".format(env_name, local_dir)) 
    return local_dir

def get_file_location_CLIP():
    """
    Set the OS environment variable AITEMPLATE_WORK_DIR to point to an absolute
    path to a directory which has AITemplate compiled artifacts the model(s). 
    Make sure the OS user running this script has read and write permissions to 
    this directory. By default, the it will look for compiled artifacts under 
    tmp/ folder of the current working directory. 
    
    """
    
    env_name = "AITEMPLATE_WORK_DIR"
    file_name = "./tmp/CLIPTextModel/test.so"
    
    if env_name in os.environ:
        file_name = os.path.join(os.environ[env_name], 'CLIPTextModel', 'test.so')
          
    print("The value of {} is {}".format(env_name, file_name)) 
    return file_name

def get_file_location_Autoencoder():
    
    """
    Set the OS environment variable AITEMPLATE_WORK_DIR to point to an absolute
    path to a directory which has AITemplate compiled artifacts the model(s). 
    Make sure the OS user running this script has read and write permissions to 
    this directory. By default, the it will look for compiled artifacts under 
    tmp/ folder of the current working directory. 
    
    """
    
    env_name = "AITEMPLATE_WORK_DIR"
    file_name = "./tmp/AutoencoderKL/test.so"
    
    if env_name in os.environ:
        file_name = os.path.join(os.environ[env_name], 'AutoencoderKL', 'test.so')
          
    print("The value of {} is {}".format(env_name, file_name))   
    
    return file_name

def get_file_location_Unet():
    
    """
    Set the OS environment variable AITEMPLATE_WORK_DIR to point to an absolute
    path to a directory which has AITemplate compiled artifacts the model(s). 
    Make sure the OS user running this script has read and write permissions to 
    this directory. By default, the it will look for compiled artifacts under 
    tmp/ folder of the current working directory. 
    """

    env_name = "AITEMPLATE_WORK_DIR"
    file_name = "./tmp/UNet2DConditionModel/test.so"
    
    if env_name in os.environ:
        file_name = os.path.join(os.environ[env_name], 'UNet2DConditionModel', 'test.so')
          
    print("The value of {} is {}".format(env_name, file_name))  
    
    return file_name