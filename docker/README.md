# Docker image for AutoDL competition

The Docker image used for AutoCV/AutoDL challenge is created with `Dockerfile`
in this directory.

Here is the [link](https://hub.docker.com/r/evariste/autodl/) to Docker Hub.

## Build Docker image for GPU/CPU
For GPU, use
```
docker build -t evariste/autodl:gpu-latest .
```
For CPU, use
```
docker build -t evariste/autodl:cpu-latest -f Dockerfile.cpu .
```
If you have push access, you can do
```
docker push evariste/autodl:gpu-latest
```
or
```
docker push evariste/autodl:cpu-latest
```

For simplicity, two scripts are written to automate the build
```
./build.sh
```
and the release
```
./release.sh
```
