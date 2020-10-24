# Thanks @dustinfreeman for providing the script
#!/bin/bash
docker build -f docker/Dockerfile_cuda9.0 -t vid2vid:cuda9.0 .

nvidia-docker run -ti --ipc=host --shm-size 12G -v $(pwd):/home/haotian/vid2vid --workdir=/home/haotian/vid2vid vid2vid:cuda9.0 /bin/bash

# docker login

# docker tag vid2vid:cuda9.0 docker.io/haotianz/vid2vid:cuda9.0

# docker push haotianz/vid2vid:cuda9.0

