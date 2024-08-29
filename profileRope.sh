#!/bin/sh

clear
make clean
make tests/profile-rope
find . -name "*.o" –delete
make tests/profile-rope
./tests/profile-rope