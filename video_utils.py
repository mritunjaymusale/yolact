import ffmpeg

def getVideoMetadata(filename):
    probe = ffmpeg.probe(filename)
    video_stream = next(
        (stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    temp_str = str(video_stream['avg_frame_rate']).split('/')
    fps = int(int(temp_str[0])/int(temp_str[1]))
    return width, height, fps



def readVideo(input_video_name):
    return (
    ffmpeg
    .input(input_video_name)
    .output('pipe:', format='rawvideo', pix_fmt='rgb24')
    .run_async(pipe_stdout=True,quiet=True)
)

def writeVideo(output_video_name,width,height,fps):
    return (
    ffmpeg
    .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height), framerate=fps,hwaccel='cuda')
    .output(output_video_name, vcodec='h264_nvenc',preset='slow',maxrate='8M',bufsize='8M',video_bitrate='8M')
    .overwrite_output()
    .run_async(pipe_stdin=True,)
)

def getAudio(filename):
    return ffmpeg.input(filename).audio

def getVideo(filename):
    return ffmpeg.input(filename).video

    