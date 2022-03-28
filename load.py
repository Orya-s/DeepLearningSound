import torch
import torchaudio
import io
from six.moves.urllib.request import urlopen

roi_model = torch.load("models\\CNN_Model-epoch_30.pth")
roi_model.eval()

speech_array, sampling_rate = torchaudio.load("/Users/rbirger/Downloads/noa.wav", normalize=True)
transform = torchaudio.transforms.Resample(sampling_rate, 16_000)

speech_array = transform(speech_array)

print((speech_array[0]))

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model()

window = 48000
step = 32000
# dataset = load_dataset("common_voice", lang, split="train", streaming=True)
# dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=16_000))
# dataset_iter = iter(dataset)
data = []
k = 0

array = speech_array[0]
array = array[44:]
print(len(array))
pointer = 0
while pointer + window < len(array):
    curr = array[pointer:pointer + window]
    print((curr))
    tensor_data = torch.tensor([curr])
    with torch.inference_mode():
        features, _ = model(tensor_data)
        print(roi_model(features))

        # pointer += step
