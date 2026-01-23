rm -r build
mkdir build
cp my_circle.g2o ./build
cp plot_g2o.py ./build
cd build
cmake ..
make -j8