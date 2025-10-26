import torch
import torchaudio
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def normalize_mfcc(mfcc):
    mean = torch.mean(mfcc, dim=1, keepdim=True)
    std = torch.std(mfcc, dim=1, keepdim=True)
    normalized_mfcc = (mfcc - mean) / (std + 1e-8)  # Adding a small value to prevent division by zero
    return normalized_mfcc


# Load your audio file
filename = "./studio/audio/studio_a_00004.wav"

# Torchaudio
waveform, sample_rate = torchaudio.load(filename)
window_size = 1024
hop_length = 512
n_mels = 128

mfcc_torchaudio = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=39, 
                                             melkwargs={'n_fft': window_size, 'n_mels': n_mels, 'hop_length': hop_length})(waveform)

mfcc_torchaudio = normalize_mfcc(mfcc_torchaudio)
mfcc_torchaudio = mfcc_torchaudio.squeeze(0).numpy()
# Visualize MFCC
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc_torchaudio, x_axis='time', sr=sample_rate)
plt.colorbar()
plt.title('MFCC (Torchaudio)')
plt.tight_layout()

# Save the visualization to a PNG file
plt.savefig('mfcc_torchaudio-norm.png')




# Librosa
y, sr = librosa.load(filename, sr=None)

mfcc_librosa = librosa.feature.mfcc(y, sr=sr, n_mfcc=39)#, n_fft=window_size, hop_length=hop_length, n_mels=n_mels)


# Visualize MFCC
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc_librosa, x_axis='time', sr=sr)
plt.colorbar()
plt.title('MFCC (Librosa)')
plt.tight_layout()

# Save the visualization to a PNG file
plt.savefig('mfcc_librosa.png')











mfcc_torchaudio = torch.Tensor(mfcc_torchaudio[0])
# Compare the MFCCs
difference = np.abs(mfcc_torchaudio.numpy() - mfcc_librosa)
print("Difference between torchaudio and librosa MFCCs:", np.mean(difference))

mfcc_librosa_torch = torch.Tensor(mfcc_librosa)
















print(mfcc_torchaudio.shape)
print(mfcc_librosa_torch.shape)
print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
print(mfcc_torchaudio[:3, :3])
print(mfcc_librosa_torch[:3, :3])
print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
print(mfcc_torchaudio.min())
print(mfcc_librosa_torch.min())
print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
print(mfcc_torchaudio.max())
print(mfcc_librosa_torch.max())
print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
print(mfcc_torchaudio.mean())
print(mfcc_librosa_torch.mean())
print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

n_mfcc_torchaudio = torch.nn.functional.normalize(mfcc_torchaudio)
n_mfcc_librosa_torch = torch.nn.functional.normalize(mfcc_librosa_torch)

print((torch.abs(n_mfcc_torchaudio - n_mfcc_librosa_torch)).mean())

print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
print(n_mfcc_torchaudio[:3, :3])
print(n_mfcc_librosa_torch[:3, :3])
print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
print(n_mfcc_torchaudio.min())
print(n_mfcc_librosa_torch.min())
print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
print(n_mfcc_torchaudio.max())
print(n_mfcc_librosa_torch.max())
print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
print(n_mfcc_torchaudio.mean())
print(n_mfcc_librosa_torch.mean())