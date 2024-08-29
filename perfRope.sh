#!/bin/sh
clear
sudo perf record -e cycles -ag -- ./tests/profile-rope
sudo perf report > ./perfReport/RoPE_llamacpp_x86_4096v4.csv
sudo perf script > ./perfReport/RoPE_llamacpp_x86_4096v4
cd perfReport
./FlameGraph/stackcollapse-perf.pl RoPE_llamacpp_x86_4096v4 > RoPE_llamacpp_x86_4096v4.folded
./FlameGraph/flamegraph.pl RoPE_llamacpp_x86_4096v4.folded > RoPE_llamacpp_x86_4096v4.svg