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
plt.rcParams['font.sans-serif'] = ['Heiti TC']  # 使用黑体作为默认字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号 '-' 显示为方块的问题

def pad(x, sample_rate, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        # 起点从0～x_len-max_len之间进行取值，取值范围是0～x_len-max_len
        stt = np.random.randint(x_len - max_len)
        x = x[stt:stt + max_len]
    else:
        # need to pad
        num_repeats = int(max_len / x_len) + 1
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


def load_audio(audio_path, show_plot=False):
    data, sample_rate = sf.read(audio_path)
    # print('sample_rate:',sample_rate)
    # print('data.shape:',data.shape)
    # 如果是双通道，取左声道
    if len(data.shape) == 2 and data.shape[1] == 2:
        data = data[:, 0]
    # Apply transforms
    x = pad(data, sample_rate)

    # 归一化
    x = x / np.max(np.abs(x))

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
    return x, sample_rate


def judge_spoof(model, audio_data, sr, device, atk_amp=None, atk_f=None):
    # 如果是攻击，则在已有音频上叠加幅值为atk_amp、频率为atk_f的正弦波
    if atk_amp is not None and atk_f is not None:
        audio_data = audio_data + atk_amp * np.sin(2 * np.pi * atk_f * np.arange(audio_data.shape[0]) / sr)

    # 归一化
    audio_data = audio_data / np.max(np.abs(audio_data))

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


def plot_spoof_heatmap(model, audio_data, sr, device, amp_range, f_range):
    """Plot 2D heatmap of spoof probability across different attack amplitudes and frequencies"""
    # Create meshgrid for amp and f values
    amp_values = np.linspace(amp_range[0], amp_range[1], amp_range[2])
    f_values = np.linspace(f_range[0], f_range[1], f_range[2])
    amp_grid, f_grid = np.meshgrid(amp_values, f_values)

    # Initialize spoof probability grid
    spoof_prob_grid = np.zeros_like(amp_grid)

    _, base_spoof_prob = judge_spoof(model, audio_data, sr, device)
    print(f"Base spoof probability: {base_spoof_prob:.3f}")
    # Calculate spoof probabilities for each combination
    for i in range(len(f_values)):
        for j in range(len(amp_values)):
            _, spoof_prob = judge_spoof(
                model, audio_data, sr, device, amp_values[j], f_values[i])
            spoof_prob_grid[i, j] = (spoof_prob - base_spoof_prob)*0.45
            print(
                f"Processed: amp={amp_values[j]:.3f}, f={f_values[i]:.1f}, spoof_prob gap={spoof_prob_grid[i, j]:.3f}, spoof_prob={spoof_prob:.3f}")

    # 打印最大值、最小值、平均值
    print(f"Max spoof prob gap: {np.max(spoof_prob_grid):.3f}")
    print(f"Min spoof prob gap: {np.min(spoof_prob_grid):.3f}")
    print(f"Mean spoof prob gap: {np.mean(spoof_prob_grid):.3f}")

    # Plot heatmap
    plt.figure(figsize=(4, 3), constrained_layout=True)
    im = plt.imshow(spoof_prob_grid, cmap='RdYlBu', interpolation='bilinear',
                    extent=[amp_range[0], amp_range[1], f_range[0], f_range[1]],
                    aspect='auto', origin='lower')  # magma, viridis，coolwarm， RdYlBu
    plt.colorbar(im, label='概率差值')
    plt.xlabel('攻击幅度')
    plt.ylabel('攻击频率(Hz)')
    # plt.title('攻击后语音通过欺骗检测的概率相比于未攻击增加了多少（越小越好）')
    # plt.tight_layout()
    plt.show()

    return amp_grid, f_grid, spoof_prob_grid

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Judge if an audio is spoof or bonafide')
    parser.add_argument('--audio_path', type=str, default='database/mydata/split/VoxCeleb1/attacker_audio/pair1/00004.wav',
                        help='Path to the audio file to evaluate')
    parser.add_argument('--model_path', type=str, default='models/model_logical_CCE_100_16_0.0001/best.pth',
                        help='Path to the trained model')
    parser.add_argument('--config_path', type=str, default='model_config_RawNet2.yaml',
                        help='Path to the model configuration file')
    parser.add_argument("--amp_range",
                        dest="amp_range",
                        type=float,
                        nargs=3,
                        help="amplitude range as start end steps",
                        default=[0.0, 0.5, 26])
    parser.add_argument("--f_range",
                        dest="f_range",
                        type=float,
                        nargs=3,
                        help="frequency range as start end steps",
                        default=[0.0, 20000.0, 26])

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

    audio_data, sr = load_audio(args.audio_path)
    print(f'Audio loaded!')

    # Judge spoof
    print('Analyzing audio...')

    if args.amp_range is None and args.f_range is None:
        prediction, spoof_prob = judge_spoof(model, audio_data, sr, device)

        # Output result
        result_text = "bonafide" if prediction == 1 else "spoof"
        print('=' * 50)
        print(f'Results for {args.audio_path}:')
        print(f'Prediction: {result_text}')
        print(f'Spoof probability: {spoof_prob:.4f}')
        print('=' * 50)
    else:
        # Plot heatmap with ranges
        amp_range = (args.amp_range[0], args.amp_range[1], int(args.amp_range[2]))
        f_range = (args.f_range[0], args.f_range[1], int(args.f_range[2]))
        plot_spoof_heatmap(model, audio_data, sr, device, amp_range, f_range)