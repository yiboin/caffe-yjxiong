rm build -rf
mkdir build && cd build
cmake .. -DUSE_MPI=ON
make -j8 && make install
cd ../
