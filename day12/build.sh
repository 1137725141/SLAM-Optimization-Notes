rm -r build
mkdir build
cp my_circle.g2o ./build
cp sphere_after.g2o ./build
cd build
cmake ..
make -j8