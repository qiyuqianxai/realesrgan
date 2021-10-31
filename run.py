import argparse
import cv2
import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
import json
from gfpgan import GFPGANer
from realesrgan import RealESRGANer
from time import sleep
# custom 1:
message_json = "/workspace/go_proj/src/Ai_WebServer/algorithm_utils/realesrgan/message.json"
user_img_dir = "/workspace/go_proj/src/Ai_WebServer/static/algorithm/realesrgan/user_imgs"
res_img_dir = "/workspace/go_proj/src/Ai_WebServer/static/algorithm/realesrgan/result_imgs"
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
            save_path = os.path.join(args.output, f'{imgname}_{model_name}_{face_enhance_str}.png')
            cv2.imwrite(save_path, output)
            print("complete！")

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

if __name__ == '__main__':
    last_msg = {}
    upsampler, face_enhancer = get_model("RealESRGAN_x4plus.pth")
    while True:
        with open(message_json, "r", encoding="utf-8") as f:
            message = json.load(f)
        if message == last_msg:
            print("wait...")
            sleep(1)
            continue
        # load parameters
        model_name = message["model"]
        face_enhance = message["face_enhance"]
        input_img = message["user_img"]

        args.input = os.path.join(user_img_dir,input_img)
        args.output = res_img_dir
        os.makedirs(args.output, exist_ok=True)
        args.face_enhance = face_enhance

        imgname, extension = os.path.splitext(os.path.basename(args.input))
        face_enhance_str = "1" if face_enhance else "0"
        res_path = os.path.join(args.output, f'{imgname}_{model_name}_{face_enhance_str}.png')
        if os.path.exists(res_path):
            continue
        # model update?
        if last_msg!={} and (last_msg["model"] != model_name):
            del upsampler,face_enhancer
            upsampler, face_enhancer = get_model(model_name)
        # inference
        run_RealEsrgan()

        last_msg = message

