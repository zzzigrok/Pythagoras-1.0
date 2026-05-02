import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
import os
import datetime # Добавлено для фиксации времени

# --- 1. АППАРАТНАЯ ОПТИМИЗАЦИЯ ПОД INTEL METEOR LAKE ---
device = 'cpu'
device_type = 'cpu'

SKY = "\033[38;5;117m"
RESET = "\033[0m"

if hasattr(torch, "xpu") and torch.xpu.is_available():
    device = 'xpu'
    device_type = 'xpu'
else:
    torch.set_num_threads(6)

# --- 2. НАСТРОЙКИ АРХИТЕКТУРЫ (СИНХРОНИЗИРОВАНО С v2.0) ---
n_embd = 256
n_head = 8
n_layer = 6
block_size = 64
dropout = 0.1

# --- 3. ОПИСАНИЕ МОДЕЛИ (СТРОГО КАК В TRAIN_MATH.PY) ---
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(n_embd, dim=2)
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class SimpleLLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        x = self.token_emb(idx) + self.pos_emb(torch.arange(T, device=device))
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.lm_head(x)

# --- 4. ЗАГРУЗКА ---
print(SKY + "Загрузка Пифагора v2.0..." + RESET)

with open('math_vocab.pkl', 'rb') as f:
    vocab_data = pickle.load(f)
stoi, itos, vocab_size = vocab_data['stoi'], vocab_data['itos'], vocab_data['vocab_size']

encode = lambda s: [stoi.get(c, 0) for c in s]
decode = lambda l: ''.join([itos.get(i, '?') for i in l])

model = SimpleLLM(vocab_size).to(device)
try:
    model.load_state_dict(torch.load('math_model_weights.pth', map_location=device, weights_only=True))
except Exception as e:
    print(f"ОШИБКА СИНХРОНИЗАЦИИ: {e}")
    exit()
model.eval()

# --- 5. ЦИКЛ ВЫЧИСЛЕНИЙ ---
print(SKY + r"""
  _____        _   _                                   
 |  __ \      | | | |                                  
 | |__) |   _ | |_| |__   __ _  __ _  ___  _ __ __ _ ___ 
 |  ___/ | | || __| '_ \ / _` |/ _` |/ _ \| '__/ _` / __|
 | |   | |_| || |_| | | | (_| | (_| | (_) | | | (_| \__ \
 |_|    \__, | \__|_| |_|\__,_|\__, |\___/|_|  \__,_|___/
         __/ |                  __/ |                    
        |___/                  |___/   [ v2.0 READY ]
""" + RESET)

while True:
    prompt = input("\nПример (например 2+2=): ").strip()
    if prompt.lower() == 'exit': break
    if not prompt: continue

    # Кодируем вход
    context = torch.tensor((encode(prompt),), dtype=torch.long, device=device)
    generated_idx = context.tolist()[0]
    
    for _ in range(15):
        # Формируем тензор [1, T] для модели
        idx_cond = torch.tensor((generated_idx[-block_size:],), dtype=torch.long, device=device)
        
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits = model(idx_cond)
            
            # Предсказываем следующий символ (жадный выбор)
            idx_next = torch.argmax(logits[0, -1, :]).item()
            generated_idx.append(idx_next)
            
            if itos.get(idx_next) == '\n': break
                
    result = decode(generated_idx).replace('\n', '').strip()
    print(f"Пифагор: {result}")
    
    # --- СОХРАНЕНИЕ В ФАЙЛ ---
    with open("math_chat_history.txt", "a", encoding="utf-8") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] Вы: {prompt}\n")
        f.write(f"[{timestamp}] Пифагор: {result}\n")
        f.write("-" * 50 + "\n\n")