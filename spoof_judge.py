import torch
from torch import Tensor
import numpy as np
import soundfile as sf
import yaml
import argparse
import os
from model import RawNet
import librosa
import matplotlib.pyplot as plt

def pad(x, sample_rate, max_len=64600, atk_amp=None, atk_f=None, show_plot=True):
    x_len = x.shape[0]
    if x_len >= max_len:
        # 起点从0～x_len-max_len之间进行取值，取值范围是0～x_len-max_len
        stt = np.random.randint(x_len - max_len)
        x = x[stt:stt + max_len]
    else:
        # need to pad
        num_repeats = int(max_len / x_len)+1
        x = np.tile(x, (1, num_repeats))[:, :max_len][0]


    
    return x




def load_model(model_path, config_path, device):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Create model
    model = RawNet(config['model'], device).to(device)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model


def load_audio(audio_path, atk_amp=None, atk_f=None, show_plot=True):
    data, sample_rate = sf.read(audio_path)
    # print('samplerate:',samplerate)
    # Apply transforms
    x = pad(data, sample_rate)

    # 归一化
    x = x / np.max(np.abs(x))

    # 如果是攻击，则在已有音频上叠加幅值为atk_amp、频率为atk_f的正弦波
    if atk_amp is not None and atk_f is not None:
        x = x + atk_amp * np.sin(2 * np.pi * atk_f * np.arange(x.shape[0]) / sample_rate)

    if show_plot:
        # 绘制音频波形图和时间频谱图在同一张图上
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        librosa.display.waveshow(x, sr=sample_rate, ax=ax1)
        ax1.set_title('Waveform')
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(x)), ref=np.max),
                                 sr=sample_rate, x_axis='time', y_axis='log', ax=ax2)
        ax2.set_title('Spectrogram')
        plt.tight_layout()
        plt.show()
    return x


def judge_spoof(model, audio_data, device):
    tensor_data = Tensor(audio_data)
    # # 打印输入数据的shape
    # print('audio_data.shape:',tensor_data.shape)
    
    # Add batch dimension and move to device
    tensor_data = tensor_data.unsqueeze(0).to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(tensor_data, None, is_test=True)
        # print('output:',output)
        prediction = torch.argmax(output, dim=1).item()
        spoof_prob = output[0][0].item()  # Probability of being spoof
    
    return prediction, spoof_prob


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Judge if an audio is spoof or bonafide')
    parser.add_argument('--audio_path', type=str, default='database/LA/ASVspoof2019_LA_eval/flac/LA_E_6720790.flac',
                        help='Path to the audio file to evaluate')
    parser.add_argument('--model_path', type=str, default='models/model_logical_CCE_100_16_0.0001/best.pth',
                        help='Path to the trained model')
    parser.add_argument('--config_path', type=str, default='model_config_RawNet2.yaml',
                        help='Path to the model configuration file')
    
    args = parser.parse_args()
    
    # Set device
    device = 'mps' if torch.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    
    # Load model
    print('Loading model...')
    model = load_model(args.model_path, args.config_path, device)
    print('Model loaded successfully!')
    
    # Load audio
    print('Loading audio...')
    if not os.path.exists(args.audio_path):
        print(f'Error: Audio file {args.audio_path} does not exist')
        exit(1)
        
    audio_data = load_audio(args.audio_path)
    print(f'Audio loaded!')
    
    # Judge spoof
    print('Analyzing audio...')
    prediction, spoof_prob = judge_spoof(model, audio_data, device)
    
    # Output result
    result_text = "bonafide" if prediction == 1 else "spoof"
    print('='*50)
    print(f'Results for {args.audio_path}:')
    print(f'Prediction: {result_text}')
    print(f'Spoof probability: {spoof_prob:.4f}')
    print('='*50)