import argparse
import functools
import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from utils.utils import print_arguments, add_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("audio_path", type=str, default="TCST/wav/Amdo/bodad/bodad_004.wav",
        help="预测的音频路径")
add_arg("model_path", type=str, default="ti_small/whisper-small-finetune",
        help="合并模型的路径，或者是huggingface上模型的名称")
add_arg("language",   type=str, default="Tibetan",
        help="设置语言，可全称也可简写，如果为None则预测的是多语言")
add_arg("task",       type=str, default="transcribe", choices=['transcribe', 'translate'],
        help="模型的任务")
add_arg("local_files_only", type=bool, default=True,
        help="是否只在本地加载模型，不尝试下载")
args = parser.parse_args()
print_arguments(args)

processor = WhisperProcessor.from_pretrained(
    args.model_path,
    language=args.language,
    task=args.task,
    local_files_only=args.local_files_only
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = WhisperForConditionalGeneration.from_pretrained(
    args.model_path,
    device_map="auto", 
    local_files_only=args.local_files_only
)

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []  
model.config.begin_suppress_tokens = []
if device == "cuda":
    model = model.half() 
else:
    model = model.float()
model.eval()


sample, sr = librosa.load(args.audio_path, sr=16000)
duration = sample.shape[-1] / sr
assert duration < 30, f"本程序只适合推理小于30秒的音频，当前音频{duration:.2f}秒，请使用其他推理程序!"

inputs = processor(
    sample,
    sampling_rate=sr,
    return_tensors="pt",
    do_normalize=True,
    return_attention_mask=True 
).to(device)  


with torch.no_grad():
    predicted_ids = model.generate(
        **inputs, 
        max_new_tokens=256
    )


transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print(f"\n识别结果：{transcription}")