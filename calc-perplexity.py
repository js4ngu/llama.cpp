import os
import subprocess
from datasets import load_dataset

def calculate_perplexity(model_path, text_chunk):
    """주어진 텍스트 청크로 퍼플렉시티를 계산."""
    command = [
        "./llama-perplexity",  # 퍼플렉시티 실행 파일
        "-m", model_path,      # 모델 경로
        "-p", text_chunk       # 텍스트 청크
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    
    # 디버깅용 출력
    print(f"Command: {' '.join(command)}")
    print(f"Return Code: {result.returncode}")
    print(f"STDOUT: {result.stdout}")
    print(f"STDERR: {result.stderr}")

    if result.returncode == 0:
        for line in result.stdout.splitlines():
            # 다양한 퍼플렉시티 출력 형태 처리
            if "perplexity:" in line:
                return float(line.split("perplexity:")[1].strip())
            elif "Final estimate: PPL =" in line:
                return float(line.split("PPL =")[1].split()[0])
    else:
        print(f"Error: {result.stderr}")
    return None

def preprocess_text(text, window_size=1024, stride=512):
    """텍스트를 슬라이딩 윈도우 방식으로 나눔."""
    tokens = text.split()
    windows = [
        " ".join(tokens[i:i+window_size])
        for i in range(0, len(tokens) - window_size + 1, stride)
    ]
    return windows

def main():
    # 모델 파일 경로
    model_path = "./Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"

    # WikiText-2 데이터셋 로드
    print("Loading dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    test_data = dataset["test"]["text"]

    # 텍스트 데이터 결합
    print("Preprocessing text...")
    full_text = " ".join([line.strip() for line in test_data if line.strip()]).replace("\n", " ")

    # 슬라이딩 윈도우 적용
    window_size = 1024
    stride = 512
    text_chunks = preprocess_text(full_text, window_size, stride)
    print(f"Generated {len(text_chunks)} chunks.")

    # 설정: 수행할 청크 수 제한
    max_chunks_to_process = 2  # 원하는 청크 수로 변경
    text_chunks = text_chunks[:max_chunks_to_process]
    print(f"Processing up to {max_chunks_to_process} chunks...")

    # 퍼플렉시티 계산
    perplexities = []
    for idx, chunk in enumerate(text_chunks):
        print(f"Processing chunk {idx + 1}/{len(text_chunks)}...")
        perplexity = calculate_perplexity(model_path, chunk)
        if perplexity is not None:
            perplexities.append(perplexity)
        else:
            print(f"Skipping chunk {idx + 1} due to calculation failure.")

    # 평균 퍼플렉시티 출력
    if perplexities:
        avg_perplexity = sum(perplexities) / len(perplexities)
        print(f"Average Perplexity: {avg_perplexity}")
    else:
        print("No perplexities were calculated successfully.")

if __name__ == "__main__":
    main()
