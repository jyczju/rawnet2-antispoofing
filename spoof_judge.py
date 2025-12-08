import torch
from torch import Tensor
import numpy as np
import soundfile as sf
import yaml
import argparse
import os
from model import RawNet
from torchvision import transforms


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    
    return padded_x


def init_transforms():
    transforms = transforms.Compose([
        lambda x: pad(x),
        lambda x: Tensor(x)
    ])
    return transforms


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


def load_audio(audio_path):
    data, samplerate = sf.read(audio_path)
    return data


def judge_spoof(model, audio_data, device):
    # Apply transforms
    transformed_data = pad(audio_data)
    tensor_data = Tensor(transformed_data)
    
    # Add batch dimension and move to device
    tensor_data = tensor_data.unsqueeze(0).to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(tensor_data, None, is_test=True)
        prediction = torch.argmax(output, dim=1).item()
        spoof_prob = output[0][1].item()  # Probability of being spoof
    
    return prediction, spoof_prob


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Judge if an audio is spoof or bonafide')
    parser.add_argument('--audio_path', type=str, required=True, 
                        help='Path to the audio file to evaluate')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model')
    parser.add_argument('--config_path', type=str, default='model_config_RawNet2.yaml',
                        help='Path to the model configuration file')
    
    args = parser.parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
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
    print(f'Audio loaded! Length: {len(audio_data)} samples')
    
    # Judge spoof
    print('Analyzing audio...')
    prediction, spoof_prob = judge_spoof(model, audio_data, device)
    
    # Output result
    result_text = "spoof" if prediction == 1 else "bonafide"
    print('='*50)
    print(f'Results for {args.audio_path}:')
    print(f'Prediction: {result_text}')
    print(f'Spoof probability: {spoof_prob:.4f}')
    print('='*50)