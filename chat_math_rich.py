import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
import os
import datetime
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.align import Align
from rich.status import Status
from rich import print as rprint

# --- 1. НАСТРОЙКИ ---
device = 'cpu'
device_type = 'cpu'
if hasattr(torch, "xpu") and torch.xpu.is_available():
    device = 'xpu'
    device_type = 'xpu'
else:
    torch.set_num_threads(6)

n_embd = 256
n_head = 8
n_layer = 6
block_size = 64
dropout = 0.1

console = Console()

# --- 2. АРХИТЕКТУРА ---
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

# --- 3. UI КОМПОНЕНТЫ ---
def print_banner():
    banner = r"""
  _____        _   _                                   
 |  __ \      | | | |                                  
 | |__) |   _ | |_| |__   __ _  __ _  ___  _ __ __ _ ___ 
 |  ___/ | | || __| '_ \ / _` |/ _` |/ _ \| '__/ _` / __|
 | |   | |_| || |_| | | | (_| | (_| | (_) | | | (_| \__ \
 |_|    \__, | \__|_| |_|\__,_|\__, |\___/|_|  \__,_|___/
         __/ |                  __/ |                    
        |___/                  |___/   [ v2.0 ]
"""
    console.print(Panel(Align.center(banner, vertical="middle"), style="bold cyan", title="[bold white]MATHEMATICAL AI[/]", subtitle="[bold white]Powered by Gemini CLI[/]"))

def load_model():
    with Status("[bold green]Загрузка Пифагора v2.0...", console=console) as status:
        if not os.path.exists('math_vocab.pkl') or not os.path.exists('math_model_weights.pth'):
            console.print("[bold red]Ошибка: Файлы модели не найдены![/]")
            return None, None, None, None, None

        with open('math_vocab.pkl', 'rb') as f:
            vocab_data = pickle.load(f)
        stoi, itos, vocab_size = vocab_data['stoi'], vocab_data['itos'], vocab_data['vocab_size']

        model = SimpleLLM(vocab_size).to(device)
        model.load_state_dict(torch.load('math_model_weights.pth', map_location=device, weights_only=True))
        model.eval()
        
        status.update("[bold blue]Модель готова к вычислениям!")
        return model, stoi, itos, vocab_size, device

# --- 4. ОСНОВНОЙ ЦИКЛ ---
def main():
    print_banner()
    model, stoi, itos, vocab_size, dev = load_model()
    if not model: return

    encode = lambda s: [stoi.get(c, 0) for c in s]
    decode = lambda l: ''.join([itos.get(i, '?') for i in l])

    console.print(f"\n[bold yellow]Информация о системе:[/]")
    table = Table(show_header=False, box=None)
    table.add_row("Устройство:", f"[bold cyan]{dev.upper()}[/]")
    table.add_row("Параметры:", "[bold cyan]~4.5M[/]")
    table.add_row("Версия:", "[bold cyan]2.0 (Rich Edition)[/]")
    console.print(table)
    
    console.print("\n[bold green]Введите 'exit' для выхода.[/]")

    while True:
        prompt = Prompt.ask("\n[bold white]Пример[/] (например 2+2=)")
        
        if prompt.lower() == 'exit':
            console.print("[bold red]Прощай, математик![/]")
            break
        if not prompt: continue
        if '=' not in prompt: prompt += '='

        # Кодируем вход
        context = torch.tensor((encode(prompt),), dtype=torch.long, device=device)
        generated_idx = context.tolist()[0]
        
        with console.status("[italic]Думаю...[/]", spinner="dots"):
            for _ in range(15):
                idx_cond = torch.tensor((generated_idx[-block_size:],), dtype=torch.long, device=device)
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits = model(idx_cond)
                    idx_next = torch.argmax(logits[0, -1, :]).item()
                    generated_idx.append(idx_next)
                    if itos.get(idx_next) == '\n': break
                    
        result = decode(generated_idx).replace('\n', '').strip()
        
        # Вывод результата в красивой панели
        res_panel = Panel(
            Align.center(f"[bold green]{result}[/]", vertical="middle"),
            title="Результат",
            border_style="bright_blue",
            expand=False
        )
        console.print(res_panel)
        
        # Сохранение истории
        try:
            with open("math_chat_history.txt", "a", encoding="utf-8") as f:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] Вы: {prompt}\n")
                f.write(f"[{timestamp}] Пифагор: {result}\n")
                f.write("-" * 50 + "\n\n")
        except:
            pass

if __name__ == "__main__":
    main()