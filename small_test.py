import argparse
import functools
import librosa
import torch  # 新增：用于判断设备和处理精度
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

# 1. 加载Whisper处理器（显式指定语言/任务，避免自动检测）
processor = WhisperProcessor.from_pretrained(
    args.model_path,
    language=args.language,
    task=args.task,
    local_files_only=args.local_files_only
)

# 2. 自动判断设备（CPU/GPU），只加载一次模型（修复重复加载问题）
device = "cuda" if torch.cuda.is_available() else "cpu"
model = WhisperForConditionalGeneration.from_pretrained(
    args.model_path,
    device_map="auto",  # 自动分配设备（GPU优先，无则用CPU）
    local_files_only=args.local_files_only
)

# 3. 清除冲突配置 + 适配设备精度（CPU不支持half()）
model.config.forced_decoder_ids = None  # 关键：消除forced_decoder_ids冲突
model.config.suppress_tokens = []       # 消除重复日志处理器警告
model.config.begin_suppress_tokens = []
if device == "cuda":
    model = model.half()  # GPU用半精度，节省内存
else:
    model = model.float() # CPU用全精度，避免不兼容
model.eval()  # 切换到评估模式，禁用训练相关功能

# 4. 读取并验证音频（保持原逻辑）
sample, sr = librosa.load(args.audio_path, sr=16000)
duration = sample.shape[-1] / sr
assert duration < 30, f"本程序只适合推理小于30秒的音频，当前音频{duration:.2f}秒，请使用其他推理程序!"

# 5. 预处理音频（新增return_attention_mask，消除注意力掩码警告）
inputs = processor(
    sample,
    sampling_rate=sr,
    return_tensors="pt",
    do_normalize=True,
    return_attention_mask=True  # 显式返回掩码，避免警告
).to(device)  # 直接将数据移到目标设备

# 6. 推理（移除forced_decoder_ids，用inputs自动传递掩码）
with torch.no_grad():  # 禁用梯度计算，节省内存+加速
    predicted_ids = model.generate(
        **inputs,  # 自动传递input_features和attention_mask
        max_new_tokens=256  # 控制生成结果长度
    )

# 7. 解码结果（保持原逻辑）
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print(f"\n识别结果：{transcription}")