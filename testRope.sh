#!/bin/sh

clear
make clean
make tests/test-rope
find . -name "*.o" –delete
make tests/test-rope
./tests/test-rope
