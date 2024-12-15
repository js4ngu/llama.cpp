conda activate wikitext_env
make clean
make llama-perplexity
ls -l | grep llama-perplexity
./llama-perplexity -m ./Meta-Llama-3-8B-Instruct-Q4_K_M.gguf -f input.txt
python /home/jongsang/llama.cpp/calc-perplexity.py