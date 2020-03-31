import subprocess
import os

Class ReCombine:
    def combine_both(output_path,input_path):
        audio_name = input_path.split('.')[0] + '.m4a'
        subprocess.call('ffmpeg -i ' + input_path + ' -vn -y -acodec copy ' + audio_name, shell=True )
        print('!!!Extract the audio {} completed!'.format(audio_name))

        video_name = output_path.split('.')[0]+'f.mp4'
        subprocess.call('ffmpeg -i ' + output_path + ' -i '+ audio_name +' -vcodec copy -acodec copy ' + video_name, shell=True)
        print('Congrats! You have generate a new video at {}!'.format(video_name))
        os.remove(audio_name)
        print('!!!Delete the audio file!')
        os.remove(output_path)
        print('!!!Delete the video file!')



