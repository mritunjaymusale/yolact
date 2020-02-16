sudo add-apt-repository ppa:savoury1/ffmpeg4 -y
sudo add-apt-repository ppa:savoury1/graphics -y
sudo add-apt-repository ppa:savoury1/multimedia -y


sudo aptitude update
sudo pip3 install cython wheel
sudo pip3 install opencv-python pillow pycocotools matplotlib ffmpeg-python pytest pytest-cov
cd /content/yolact/external/DCNv2
sudo python3 setup.py build develop
cd /content
git clone https://github.com/chentinghao/download_google_drive.git
mkdir -p /content/yolact/weights
python /content/download_google_drive/download_gdrive.py 1ZPu1YR2UzGHQD0o1rEqy-j5bmEm3lbyP /content/yolact/weights/yolact_plus_resnet50_54_800000.pth

cd /content
sudo aptitude install -y libavformat-dev libavcodec-dev libavfilter-dev libavutil-dev ffmpeg libx264-dev libavdevice-dev   libswscale-dev libswresample-dev





 
 




