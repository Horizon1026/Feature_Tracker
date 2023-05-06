if [ ! -d "build" ]; then
    mkdir build
fi

cd build/
rm * -rf
cmake -G "MinGW Makefiles" .. -DOpenCV_DIR="E:\\OpenCV4\\opencv\\mingw64_build"
mingw32-make.exe -j
cd ..