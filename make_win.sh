if [ -d "build" ]; then
    cd build/
    mingw32-make.exe -j
    cd ..
fi
