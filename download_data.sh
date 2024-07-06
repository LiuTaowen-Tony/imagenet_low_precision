#! /bin/sh

[ $# = 1 ] || (echo "usage : $0 output_dir" && exit)


wget -c https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar  -P $1
wget -c https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar -P $1
wget -c https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar -P $1

