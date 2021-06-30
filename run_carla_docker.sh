docker run -p 2000-2002:2000-2002 --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 carlasim/carla:0.9.9 /bin/bash CarlaUE4.sh -opengl –carla-server -benchmark -fps=10
