##################################
# 작동을 안함, 근데 왜인지는 모르겠음.
##################################


import time
import torch
from transformers import LlamaForCausalLM, AutoTokenizer

# GPU 사용 여부 확인
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    device = torch.device("cuda")
else:
    print("GPU is not available, using CPU.")
    device = torch.device("cpu")

# 모델 경로 설정
model_path = "Assets/Models/llama_3_1_8B_instruct"

# 모델과 토크나이저 초기화
model = LlamaForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,  # LLaMA 3.1에 필요한 옵션
    load_in_8bit=False,      # 8비트 양자화 비활성화
)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True  # LLaMA 3.1에 필요한 옵션
)

def generate_response(input_text, max_length=500, temperature=0.7):
    # 입력 텍스트 토크나이징
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    # 시작 시간 기록
    start_time = time.time()
    
    # 모델 출력 생성
    with torch.no_grad():  # 추론 시 메모리 사용량 감소
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,  # 패딩 토큰 명시적 지정
            repetition_penalty=1.2,               # 반복 방지
        )
    
    # 출력 디코딩
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 종료 시간 기록 및 계산
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    
    return generated_text

# 예시 사용
input_text = "Why is the arm of T-rex so short??"
response = generate_response(input_text)

# 결과를 파일로 저장
with open("output.txt", "w", encoding="utf-8") as text_file:
    text_file.write(response)

print(response)