# DEM方法与退化生物材料计算结合：生物材料退化的模型
参考文献:
- [1] Rolf-Pissarczyk, M., Li, K., Fleischmann, D., & Holzapfel, G. A. (2021). A discrete approach for modeling degraded elastic fibers in aortic dissection. Computer Methods in Applied Mechanics and Engineering, 373, 113511. https://doi.org/10.1016/j.cma.2020.113511
- [2] Holzapfel, G. A., Gasser, T. C., & Ogden, R. W. (2000). A new constitutive framework for arterial wall mechanics and a comparative study of material models. Journal of Elasticity, 61(1–3), 1–48. https://doi.org/10.1023/A:1010835316564

当前问题：
- 文章[1]中的材料体积模量 $K=\mathrm{60e3}$， $\mu=\mathrm{62.1e3}$, 是否正确？纤维的积分只考了了弹性纤维，未考虑蛋白质纤维。
- 纤维积分点个数为 `5 * 20 = 100`， 是否足够?
- 如何通过Fenics求出对比的解？
- 如何施加随机破坏参数场？
- 如何设计numerical examples，用于展示肌肉纤维的方向的影响、破坏参数的影响。


--------------------------------------------------------------------
Paper: 
A deep energy method for finite deformation hyperelasticity

Authors: Vien Minh Nguyen-Thanh, Xiaoying Zhuang, Timon Rabczuk

European Journal of Mechanics - A/Solids
Available online 25 October 2019, 103874
https://doi.org/10.1016/j.euromechsol.2019.103874

Contact: ntvminh286@gmail.com (institute email: minh.nguyen@iop.uni-hannover.de)

![](loss.gif)

![](Tbar-uncon.gif)

--------------------------------------------------------------------
Setup:
1. Create DeepEnergyMethod directory: cd \<workingdir\>; mkdir DeepEnergyMethod

2. Download dem_hyperelasticity source code and put it under DeepEnergyMethod.
The directory is like \<workingdir\>/DeepEnergyMethod/dem_hyperelasticity

1. Setup environment with conda: conda create -n demhyper python=3.7

2. Switch to demhyper environment to start working with dem: source activate demhyper

3. Install numpy, scipy, matplotlib: pip install numpy scipy matplotlib

4. Install pytorch and its dependencies: conda install pytorch; conda install pytorch-cpu torchvision-cpu -c pytorch

5. Install pyevtk for view in Paraview: pip install pyevtk

6. Setup PYTHONPATH environment by doing either a.(temporary use) or b.(permanent use): 

a. export PYTHONPATH="$PYTHONPATH:\<workingdir\>/DeepEnergyMethod"
  
b. add the above line to the end of file ~/.bashrc and execute "source ~/.bashrc"

Optional:

To use fem to compare the results with fem, we recommend to install fenics

1. conda config --add channels conda-forge

2. !conda install fenics
