# CARLA Tutorial

How to install CARLA on an Ubuntu server, run it headless, and compute stuff using Python and Detectron2

## Installing CARLA

There are various ways to install CARLA:
- Easiest: Extract pre-computed binary file from their Github repo: https://github.com/carla-simulator/carla/blob/master/Docs/download.md. 
Personally, on Ubuntu 16.04, 18.04 and 20.04, I have found version 9.10 and above to be unstable. Version 0.9.9 works perfectly for me without any compromise for features. 
If you use the stable version - 0.8.2, you lose out on a lot of cool customization features. 
    - Extract the package. 
    - If you are on a Ubuntu machine with a display and just want to play around: run `./CarlaUE4.sh`.
    - If you do not have a display (remote server), and want to use the CARLA with a PythonAPI: run `DISPLAY= ./CarlaUE4.sh -opengl -carla-server -benchmark -fps=10` on the command line.
    - This will always run on GPU 0. If you need to run on another GPU, see below. 

- **Second Easiest and Recommended: Use Docker + NVIDIA-Docker**:
    - First, install Docker CE 20.10.5 (do not install 20.10.6, it will require IPv6 to be enabled, or will throw an error) and NVIDIA-Docker2. This needs CUDA>10. Follow instructions in `install_docker.sh` and `install_nvidia_docker.sh` in this repo.  
    - After instaling, pull the carla image from dockerhub:
        ```
        docker pull carlasim/carla:0.9.9
        docker run -p 2000-2002:2000-2002 --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 carlasim/carla:0.9.9 /bin/bash CarlaUE4.sh -opengl –carla-server -benchmark -fps=10
        ```
    - Use the `NVIDIA_VISIBLE_DEVICES=0` flag to choose your GPU number. 
    - This should print stuff like:
        ```
        4.24.3-0+++UE4+Release-4.24 518 0
        Disabling core dumps.
        sh: 1: xdg-user-dir: not found
        ```
        This is fine, it means carla is running. To verify, run `docker container ls` and it should show a carla container running. If you don't see anything, then it must have failed silently. Look at steps below.  
    - Sometimes, the above command can give a segmentation fault with signal 11 and memory overflow (sometimes it fails silently). This usually means the GPU is busy.
    - To double check, run the CARLA container in interactve mode and check the GPU:
        ```
        docker run -p 2000-2002:2000-2002 --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 -it carlasim/carla:0.9.9 /bin/bash​
        nvidia-smi
        ```
    - Make sure there is only 1 GPU shown and it's empty. 
    - From inside and container, run `./CarlaUE4.sh -opengl -carla-server -benchmark -fps=10`
    - If it still doesn't work, you will probably need to try another version of CARLA - most likely a lower version. 
    - Another option is to try the stable 0.8.2 version. It lacks functionality, but it has a verbose output when running the `./CarlaUE4.sh` file. This can give you an idea of what might be going wrong. 
    

- Hard: Building from source. Needs a bunch of sudo commands and only do it if all else fails. Haven't tried this method. 

## Connecting the Python Client. 
Once CARLA is running somehow using either of the two options above, it's time to now connect a Python Client to control stuff. 

If you are using the Docker installation approach:

1. Make a folder `mkdir CARLA_docker` or whatever you want to name it. 

2. Copy the PythonAPI from the docker container that is running to this folder. 
```
cd CARLA_docker
docker cp <carla-container-id>:/home/carla/PythonAPI .
```

You can get the `<carla-container-id>` by doing a `docker container ls`

This is very important. Do not use the PythonAPI/ folder from any other version or even the same version's tar precomputed binary without docker. It must come from the same docker version of CARLA. Otherwise, it will throw an error like `rpc::rpc_error during call in function version​`

3. Note that if you are using stable 0.8.2, the `PythonAPI/` folder was called `PythonClient/`. Follow their code in the examples/ folder. It's simple, but lacks functionality. I will focus on `PythonAPI/` found in version 0.9.9.   

Now, in this repository, go to open `playground.ipynb` on a jupyter noteboook and follow along for a PythonAPI tutorial. 



