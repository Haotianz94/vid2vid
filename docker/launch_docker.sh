# Thanks @dustinfreeman for providing the script
#!/bin/bash
nvidia-docker build -f docker/Dockerfile -t vid2vid:CUDA9-py35 .

nvidia-docker run -ti --ipc=host --shm-size 32G -v $(pwd):/mnt_sora --workdir=/mnt_sora haotianz/vid2vid:CUDA9-py35 /bin/bash

docker login

docker tag vid2vid:CUDA9-py35 docker.io/haotianz/vid2vid:CUDA9-py35

docker push haotianz/vid2vid:CUDA9-py35

