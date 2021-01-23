sudo add-apt-repository ppa:savoury1/ffmpeg4 -y
#sudo add-apt-repository ppa:savoury1/graphics -y
sudo add-apt-repository ppa:savoury1/multimedia -y

sudo apt install aptitude -y
sudo aptitude update
sudo pip3 install cython wheel
sudo pip3 install opencv-python pillow pycocotools matplotlib ffmpeg-python pytest pytest-cov pylint rope filterpy==1.4.1
cd /content/yolact/external/DCNv2
sudo python3 setup.py build develop
cd /content
git clone https://github.com/chentinghao/download_google_drive.git
mkdir -p /content/yolact/weights
python /content/download_google_drive/download_gdrive.py 1ZPu1YR2UzGHQD0o1rEqy-j5bmEm3lbyP /content/yolact/weights/yolact_plus_resnet50_54_800000.pth

cd /content
sudo aptitude install -y libavformat-dev libavcodec-dev libavfilter-dev libavutil-dev ffmpeg libx264-dev libavdevice-dev   libswscale-dev libswresample-dev

#sudo  apt-get update && sudo apt-get install -y git make yasm pkg-config

#git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers \
#  && cd nv-codec-headers \
#  && git checkout sdk/9.0 \
#  && sudo make install

#git clone https://git.ffmpeg.org/ffmpeg.git \
#  && cd ffmpeg \
#  && git checkout 018a427 \
#  && ./configure --enable-cuda --enable-cuvid --enable-nvenc --enable-nonfree --enable-libnpp --extra-cflags=-I/usr/local/cuda/include  --extra-ldflags=-L/usr/local/cuda/lib64 \
#  && sudo make -j -s && sudo cp ffmpeg /usr/local/bin && sudo cp ffprobe /usr/local/bin/


 
 




