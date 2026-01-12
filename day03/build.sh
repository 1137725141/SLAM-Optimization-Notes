rm -r build
mkdir build
cd build
cmake ..
make -j8
cp ../test_image.jpg ./