import unittest
import os
import ffmpeg


class BasicVideoTest(unittest.TestCase):

    def test_if_video_duration_match(self):
        video_stream = self.getVideoStream('tucker.mp4')
        input_video_duration = float(video_stream['duration'])

        video_stream = self.getVideoStream('final.mp4')
        output_video_duration = float(video_stream['duration'])

        self.assertEqual(output_video_duration, input_video_duration)

    def test_fps(self):
        video_stream = self.getVideoStream('tucker.mp4')
        input_video_fps = self.calculateFPS(video_stream)

        video_stream = self.getVideoStream('final.mp4')
        output_video_fps = self.calculateFPS(video_stream)

        self.assertEqual(input_video_fps, output_video_fps)

    def calculateFPS(self, video_stream):
        temp_str = str(video_stream['avg_frame_rate']).split('/')
        return int(int(temp_str[0])/int(temp_str[1]))

    def getVideoStream(self, filename):
        probe = ffmpeg.probe(filename)
        video_stream = next(
            (stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        return video_stream



if __name__ == "__main__":
    unittest.main()
