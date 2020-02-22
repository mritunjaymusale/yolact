import unittest
import os
import ffmpeg


class BasicVideoTest(unittest.TestCase):
    input_file_name='out.mp4'
    output_file_name ='final.mp4'
    def test_if_video_duration_match(self):
        video_stream = self.getVideoStream(self.input_file_name)
        input_video_duration = float(video_stream['duration'])

        video_stream = self.getVideoStream(self.output_file_name)
        output_video_duration = float(video_stream['duration'])

        self.assertEqual(output_video_duration, input_video_duration)

    def test_fps(self):
        video_stream = self.getVideoStream(self.input_file_name)
        input_video_fps = self.calculateFPS(video_stream)

        video_stream = self.getVideoStream(self.output_file_name)
        output_video_fps = self.calculateFPS(video_stream)

        self.assertEqual(input_video_fps, output_video_fps)

    def test_resolution(self,):
        video_stream = self.getVideoStream(self.input_file_name)
        input_video_width,input_video_height =self.calculateHeightAndWidth(video_stream)

        video_stream = self.getVideoStream(self.output_file_name)
        output_video_width,output_video_height =self.calculateHeightAndWidth(video_stream)
        self.assertEqual(input_video_width,output_video_width)
        self.assertEqual(input_video_height,output_video_height)

    def calculateHeightAndWidth(self, video_stream):
        video_width = int(video_stream['width'])
        video_height = int(video_stream['height'])
        return video_width,video_height



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
