# How to run pytorch with native alpaka vector add from python

```bash
# lets create a folder where we will execute all command in to keep the storage clean
mkdir letsPlay && cd letsPlay

# we will build everything 
git clone https://github.com/psychocoderHPC/alpakaTorchVAdd.git

# load python module or as in this case via spack
spack load py-pip ^python@3.11.6 %gcc@12.2.0

python3 -m venv alpakaTorch
source alpakaTorch/bin/activate

# load cuda module or as in this case via spack
spack load cuda@12.5.0

# install torch with cuda support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip3 install matplotlib

# download alpaka with small modifications to use native cuda devices and queues
git clone -b ai-pytorch2StudiProject https://github.com/psychocoderHPC/alpaka3.git
mkdir build && cd build

# configure alpaka
# you need a modern cmake version 3.22.1 or newer
cmake ../alpaka3 -DCMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)") -DCMAKE_BUILD_TYPE=Release -Dalpaka_DEP_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 -Dalpaka_EXAMPLES=ON

# build the vector add alpaka torch example only
cmake --build . --target pytorchVectorAdd

# the python examples hard code the name of the library so we need to create a symlink that th example can find it
# this must only be executed once
cd ..
cd alpakaTorchVAdd
ln -s $(pwd)/../build/example/pytorch/libpytorchVectorAdd.so vectoradd.so

# tiny test
./vectorAdd.py

# perf cuda and cpu
./vecPerf.py

# perf cuda and cpu with better plot
./vecPerf2.py

# perf cuda and cpu and plot relative speedup
./vecPerfSpeed.py
```

