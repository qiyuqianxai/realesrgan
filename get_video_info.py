import cv2

def get_source_info_opencv(source_name):
    cap = cv2.VideoCapture(source_name)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH )
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print("width:{} \nheight:{} \nfps:{} \nnum_frames:{}".format(width, height, fps, num_frames))
    return fps
import os
if __name__ == "__main__":
    # get_source_info_opencv("11.mp4")
    fps = get_source_info_opencv("/workspace/go_proj/src/Ai_WebServer/static/algorithm/realesrgan/user_videos/11.mp4")
    res_path = "11.mp4"
    os.system(f"ffmpeg -framerate {fps} -f image2 -i video_temp/%4d.jpg  -vcodec libx264 -vf fps={fps} {res_path} -y")
