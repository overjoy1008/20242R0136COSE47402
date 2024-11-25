import time
import torch

print(torch.version.cuda)

# GPU 사용 여부 확인
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    device = torch.device("cuda")
else:
    print("GPU is not available, using CPU.")
    device = torch.device("cpu")

from transformers import LlamaForCausalLM, AutoTokenizer

# 모델 경로 설정
model_path = "Assets/Models/llama_3_2_3B_instruct"

# 모델과 토크나이저 초기화
model = LlamaForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 입력 텍스트 설정
input_text = "You are an excellent dinosaur expert and a scientist.\nQuestion: Why is the arm of T-rax so short??\nAnswer:"

# 입력 텍스트 토크나이징
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# 시작 시간 기록
start_time = time.time()

# 모델 출력 생성
outputs = model.generate(
    **inputs,
    max_new_tokens=500,
    temperature=0.7,  # 생성의 창의성 조절 (0에 가까울수록 결정적)
    do_sample=True,   # 다양한 응답 생성 허용
)

# 출력 디코딩
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 종료 시간 기록
end_time = time.time()

# 소요 시간 계산 및 출력
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")

print(generated_text)

# 결과를 파일로 저장
with open("output.txt", "w", encoding="utf-8") as text_file:
    text_file.write(generated_text)