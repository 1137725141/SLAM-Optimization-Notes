rm -r build
mkdir build
cp 1.png ./build/
cp 2.png ./build/
cd build
cmake ..
make -j8