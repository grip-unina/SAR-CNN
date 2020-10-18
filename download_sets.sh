mkdir sets
cd sets

wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz
tar -xf BSR_bsds500.tgz

mkdir Train400
cp ./BSR/BSDS500/data/images/train/*.jpg Train400/
cp ./BSR/BSDS500/data/images/test/*.jpg  Train400/

mkdir Val68
xargs -I file -a ../dataset/BSDS_val68_list.txt cp ./BSR/BSDS500/data/images/val/file Val68/

rm BSR_bsds500.tgz
rm -r BSR


wget --no-check-certificate https://download.visinf.tu-darmstadt.de/data/denoising_datasets/Set12.zip
unzip Set12.zip
rm Set12.zip
