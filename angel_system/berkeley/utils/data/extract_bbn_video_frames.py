import glob
import cv2
import os

from pathlib import Path


def bbn_video_extract_images(skill):
    #bbn_root = '/media/hannah.defazio/Padlock_DT/Data/notpublic/PTG/Release_v0.5'
    bbn_root = '/data/ptg/medical/bbn/Release_v0.5/v0.52/'
    #bbn_root = '/media/hannah.defazio/Padlock_DT5/Data/notpublic/PTG/Release_v0.5/v0.52'
    skill_dir = f'{bbn_root}/{skill}/Data'
    print(skill_dir)

    for video_fn in glob.glob(f'{skill_dir}/*/*.mp4', recursive=True):
        print(video_fn)
        out_dir = Path(os.path.dirname(video_fn) + '/_extracted/images')
        out_dir.mkdir(parents=True, exist_ok=True)

        vidcap = cv2.VideoCapture(video_fn)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        success, image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(f"{out_dir}/frame_{count}_{count/fps}.png", image)     
            success, image = vidcap.read()
            count += 1

def main():
    skill = 'M2_Tourniquet'
    bbn_video_extract_images(skill)

if __name__ == '__main__':
    main()
