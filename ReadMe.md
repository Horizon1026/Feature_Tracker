# Feature tracker
Intergration of some simple sparse feature track method.

# Components
- [x] Optical flow tracker
  - [ ] Basic klt
    - [x] Direct
    - [x] Inverse
    - [x] Fast
    - [ ] SSE
    - [ ] Neon
  - [ ] Affine klt
    - [x] Direct
    - [x] Inverse
    - [x] Fast
    - [ ] SSE
    - [ ] Neon
  - [ ] Lssd klt
    - [x] Direct
    - [x] Inverse
    - [x] Fast
    - [ ] SSE
    - [ ] Neon
- [x] Direct method tracker
  - [x] Direct
  - [x] Inverse
  - [ ] Fast
- [x] Descripter matcher
  - [x] Nearby matching
  - [x] Force matching

# Dependence
- Slam_Utility
- Feature_Detector
- Visualizor2D
- OpenCV (Only to compare, not necessary)
- libtorch (nn feature match needed)

# Compile and Run
- 第三方仓库的话需要自行 apt-get install 安装
- 拉取 Dependence 中的源码，在当前 repo 中创建 build 文件夹，执行标准 cmake 过程即可
```bash
mkdir build
cmake ..
make -j
```
- 编译成功的可执行文件就在 build 中，具体有哪些可执行文件可参考 run.sh 中的列举。可以直接运行 run.sh 来依次执行所有可执行文件

```bash
sh run.sh
```

# Tips
- 欢迎一起交流学习，不同意商用；
