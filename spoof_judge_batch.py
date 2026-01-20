import torch

import numpy as np
import argparse
import os
from tqdm import tqdm
import re

from spoof_judge import load_audio, judge_spoof, load_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Judge if an audio is spoof or bonafide')
    parser.add_argument('--audio_path', type=str, default='database/mydata/volunteer/orig_audio/',
                        help='Path to the audio file to evaluate')
    parser.add_argument('--model_path', type=str, default='models/model_logical_CCE_100_16_0.0001/best.pth',
                        help='Path to the trained model')
    parser.add_argument('--config_path', type=str, default='model_config_RawNet2.yaml',
                        help='Path to the model configuration file')
    parser.add_argument('--atk', type=bool, default=False)
    
    args = parser.parse_args()
    
    # Set device
    device = 'mps' if torch.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    
    # Load model
    print('Loading model...')
    model = load_model(args.model_path, args.config_path, device)
    print('Model loaded successfully!')

    atk_amps = []
    atk_fs = []
    if args.atk:
        print("Attack")
        if "aishell" in args.audio_path:
            atk_amps = [0.0581, 0.0648, 0.036, 0.2922, 0.1546, 0.0095, 0.0573, 0.0555, 0.0436, 0.3988]
            atk_fs = [3671.06, 4592.98, 943.95, 3542.28, 4954.2, 2133, 636.12, 1440.66, 332.77, 696.97]
        if "VoxCeleb" in args.audio_path:
            atk_amps = [0.5, 0.5, 0.3966, 0.1178, 0.44, 0.5, 0.5, 0.3378, 0.5, 0.1344]
            atk_fs = [1999.99, 10000, 7060.15, 6583.37, 9498.15, 3347.5, 3100.75, 4320.05, 5000, 1074.48]

    spoof_probs = []
    spoof_flags = []
    # 遍历audio_path下的所有文件
    for file in os.listdir(args.audio_path):
        if not file.endswith(".wav"):
            continue
        # 提取文件名中的数字
        match = re.search(r'\d+', file)
        if match:
            number = int(match.group())
            print(f"File: {file}, Number: {number}")
        else:
            print(f"File: {file}, No number found")

        # 循环100次，记录平均Spoof probability
        # for i in tqdm(range(100)):
        for i in tqdm(range(143)):
            # Load audio
            if args.atk:
                audio_data = load_audio(args.audio_path + file, atk_amp=atk_amps[number - 1],
                                          atk_f=atk_fs[number - 1], show_plot=False)
            else:
                audio_data = load_audio(args.audio_path + file, show_plot=False)

            # Judge spoof
            prediction, spoof_prob = judge_spoof(model, audio_data, device)

            spoof_probs.append(spoof_prob)
            spoof_flags.append(1 if prediction == 0 else 0)
    print("Average Spoof probability: {:.4f}".format(np.mean(spoof_probs)))
    print("Average Spoof percent: {:.4f}".format(np.sum(spoof_flags) / len(spoof_flags)))