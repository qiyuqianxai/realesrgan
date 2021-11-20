import argparse
import cv2
import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
import json
import shutil
from gfpgan import GFPGANer
from realesrgan import RealESRGANer
from time import sleep

# custom 1:
message_json = "/workspace/go_proj/src/Ai_WebServer/algorithm_utils/realesrgan/message.json"
user_img_dir = "/workspace/go_proj/src/Ai_WebServer/static/algorithm/realesrgan/user_imgs"
res_img_dir = "/workspace/go_proj/src/Ai_WebServer/static/algorithm/realesrgan/result_imgs"

user_video_dir = "/workspace/go_proj/src/Ai_WebServer/static/algorithm/realesrgan/user_videos"
res_video_dir = "/workspace/go_proj/src/Ai_WebServer/static/algorithm/realesrgan/res_videos"
# message_json = "./message.json"
# user_img_dir = "./user_imgs"
# res_img_dir = "./res_imgs"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='inputs', help='Input image or folder')
    parser.add_argument(
        '--model_path',
        type=str,
        default='experiments/pretrained_models/RealESRGAN_x4plus.pth',
        help='Path to the pre-trained model')
    parser.add_argument('--output', type=str, default='results', help='Output folder')
    parser.add_argument('--netscale', type=int, default=4, help='Upsample scale factor of the network')
    parser.add_argument('--outscale', type=float, default=4, help='The final upsampling scale of the image')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored image')
    parser.add_argument('--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
    parser.add_argument('--half', action='store_true', help='Use half precision during inference')
    parser.add_argument('--block', type=int, default=23, help='num_block in RRDB')
    parser.add_argument(
        '--alpha_upsampler',
        type=str,
        default='realesrgan',
        help='The upsampler for the alpha channels. Options: realesrgan | bicubic')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
    args = parser.parse_args()
    return args
args = get_args()

def run_RealEsrgan():
    paths = [args.input]
    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))
        print('Testing', idx, imgname)

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        h, w = img.shape[0:2]
        if max(h, w) > 1000 and args.netscale == 4:
            import warnings
            warnings.warn('The input image is large, try X2 model for better performance.')
        if max(h, w) < 500 and args.netscale == 2:
            import warnings
            warnings.warn('The input image is small, try X4 model for better performance.')

        try:
            if args.face_enhance:
                _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                output, _ = upsampler.enhance(img, outscale=args.outscale)
        except Exception as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        else:
            face_enhance_str = "1" if face_enhance else "0"
            save_path = os.path.join(args.output, f'{model_name}_{face_enhance_str}_{input_img}')
            cv2.imwrite(save_path, output)
            print("complete！")

def run_RealEsrgan_video(res_path):
    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if os.path.exists("video_temp"):
        shutil.rmtree("video_temp")
    os.makedirs("video_temp")
    if cap.isOpened():
        img_num = 0
        while cap.grab():
            _, img = cap.retrieve()
            h, w = img.shape[0:2]
            if max(h, w) > 1000 and args.netscale == 4:
                import warnings
                warnings.warn('The input image is large, try X2 model for better performance.')
            if max(h, w) < 500 and args.netscale == 2:
                import warnings
                warnings.warn('The input image is small, try X4 model for better performance.')

            try:
                if args.face_enhance:
                    _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
                else:
                    output, _ = upsampler.enhance(img, outscale=args.outscale)
            except Exception as error:
                print('Error', error)
                print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
            else:
                save_path = os.path.join("video_temp","%04d.jpg"%img_num)
                cv2.imwrite(save_path, output)
                print(img_num)
                img_num += 1
        os.system(f"ffmpeg -framerate {fps} -f image2 -i video_temp/%4d.jpg  -vcodec libx264 -vf fps={fps} {res_path} -y")
        print("complete!")


def get_model(model_name):
    args.model_path = os.path.join("experiments/pretrained_models", model_name)
    if 'RealESRGAN_x4plus_anime_6B.pth' in args.model_path:
        args.block = 6
    elif 'RealESRGAN_x2plus.pth' in args.model_path:
        args.netscale = 2
    elif 'RealESRGAN_x4plus.pth' in args.model_path:
        args.netscale = 4
    print("load model...")
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=args.block, num_grow_ch=32,
                    scale=args.netscale)

    upsampler = RealESRGANer(
        scale=args.netscale,
        model_path=args.model_path,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=args.half)

    face_enhancer = GFPGANer(
        model_path='https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth',
        upscale=args.outscale,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=upsampler)


    return upsampler,face_enhancer

def handle_img():
    global upsampler, face_enhancer
    args.output = res_img_dir
    os.makedirs(args.output, exist_ok=True)
    args.face_enhance = face_enhance

    face_enhance_str = "1" if face_enhance else "0"
    res_path = os.path.join(args.output, f'{model_name}_{face_enhance_str}_{input_img}')
    print(res_path)
    if os.path.exists(res_path):
        print("RealESRGAN exists...")
        sleep(1)
        return
    # model update?
    if last_msg != {} and (last_msg["model"] != model_name):
        del upsampler, face_enhancer
        upsampler, face_enhancer = get_model(model_name)
    # inference
    run_RealEsrgan()

def handle_video():
    global upsampler, face_enhancer
    args.output = res_video_dir
    os.makedirs(args.output, exist_ok=True)
    args.face_enhance = face_enhance

    face_enhance_str = "1" if face_enhance else "0"
    res_path = os.path.join(args.output, f'{model_name}_{face_enhance_str}_{input_img}')
    print("video",res_path)
    if os.path.exists(res_path):
        print("RealESRGAN exists...")
        sleep(1)
        return
    # model update?
    if last_msg != {} and (last_msg["model"] != model_name):
        del upsampler, face_enhancer
        upsampler, face_enhancer = get_model(model_name)
    # inference
    run_RealEsrgan_video(res_path)

if __name__ == '__main__':
    last_msg = {}
    upsampler, face_enhancer = get_model("RealESRGAN_x4plus.pth")
    while True:
        try:
            with open(message_json, "r", encoding="utf-8") as f:
                message = json.load(f)
        except Exception as e:
            print(e)
            sleep(1)
            continue
        if message == last_msg:
            sleep(1)
            print("RealESRGAN wait...")
            continue
        # load parameters
        model_name = message["model"]
        face_enhance = message["face_enhance"]
        input_img = message["user_img"]
        imgname, extension = os.path.splitext(input_img)

        if extension in [".mp4"]:
            # deal video
            args.input = os.path.join(user_video_dir, input_img)
            handle_video()
        else:
            args.input = os.path.join(user_img_dir, input_img)
            handle_img()

        last_msg = message
        sleep(1)
