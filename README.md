# RL_NetHack_2020
Reinforcement Learning solution for NetHack environment

Authors: 
- Rifumo Mzimba 
- Caston Nyabadza
- Sarah Wookey

We attempt to solve the [NetHack] environment using the two different implementations listed below:

## Policy Gradient method
- Actor-Critic
## Value Function method
- Deep Q-network




## Code
The code is set up to run on colab as a jupyter notebook, to set NLE enviroment go see [NetHack]

## Install NLE
For information on how to install the enviroment locally, go see [NLE]. For Windows install, go see [NLE] for the docker install
or follow the install instructions below for instructions on how to install on linux. :

### Update cMake
cd  <br>
wget https://github.com/Kitware/CMake/releases/download/v3.18.4/cmake-3.18.4.tar.gz  <br>
tar xzvf cmake-3.18.4.tar.gz <br>
cd cmake-3.18.4/ <br>
./configure --prefix=$HOME <br>
make <br>
make install <br>


Now put this in your ~/.bashrc<br>

export PATH=$HOME/bin:$PATH<br>

Now run 

source ~/.bashrc<br>

To confirm, run the following, which should tell you the version is 3.18

cmake --version

### Install Bison

cd<br>
wget http://ftp.gnu.org/gnu/bison/bison-2.3.tar.gz<br>
tar -xvzf bison-2.3.tar.gz<br>
cd bison-2.3<br>
./configure --prefix=$HOME<br>
make<br>
make install<br>
source ~/.bashrc<br>
which bison<br>

### Install Flex


cd<br>
wget https://downloads.sourceforge.net/project/flex/flex-2.6.0.tar.gz<br>
tar xzvf flex-2.6.0.tar.gz<br>
cd flex-2.6.0/<br>
./configure --prefix=$HOME<br>
make<br>
make install<br>
source ~/.bashrc<br>
which flex<br>


### Install NLE

conda create -n nle python=3.8<br>
conda activate nle<br>
pip install --no-use-pep517 nle<br>


### Verify all is well

python<br>
import gym<br>
import nle<br>
env = gym.make("NetHackScore-v0")<br>
env.reset()<br>
env.step(1)<br>
env.render()<br>



 ## Useful NLE links
Nethack baseline Agent: <br>
https://github.com/facebookresearch/nle/blob/master/nle/agent/agent.py

NetHack Learning Environment: <br>
https://github.com/facebookresearch/nle 

NetHack Wiki:<br>
https://nethackwiki.com/

NetHack Learning Environment Research Paper: <br>
https://arxiv.org/pdf/2006.13760.pdf



[NetHack]: <https://github.com/NetHack/NetHack>
[NLE]: <https://github.com/facebookresearch/nle> 
