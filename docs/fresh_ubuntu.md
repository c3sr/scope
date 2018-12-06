# Ubuntu 18.04.1



```bash
# Install build tools
sudo apt install -y g++ make wget git

# Install CUDA
cm /tmp
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get install -y cuda-toolkit-10-0
echo 'export PATH="/usr/local/cuda/bin:$PATH"' >> ~/.bashrc
. ~/.bashrc


# Install Cmake 3.12.0 in $HOME/software/cmake-3.12.0
cd /tmp
wget https://cmake.org/files/v3.12/cmake-3.12.0-Linux-x86_64.sh
mkdir -p $HOME/software/cmake-3.12.0
sudo sh cmake-3.12.0-Linux-x86_64.sh --prefix=$HOME/software/cmake-3.12.0 --exclude-subdir
echo 'export PATH="$HOME/software/cmake-3.12.0/bin:$PATH"' >> ~/.bashrc
. ~/.bashrc

# Install SCOPE with only the Example|Scope Enabled
git clone https://github.com/c3sr/scope.git
cd scope
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release .. \
  -DENABLE_MISC=0 \
  -DENABLE_NCCL=0 \
  -DENABLE_COMM=0 \
  -DENABLE_CUDNN=0
make
```
