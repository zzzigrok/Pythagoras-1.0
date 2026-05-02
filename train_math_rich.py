import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
import time
import os
import math
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table
from rich.live import Live

# --- 1. АППАРАТНАЯ ОПТИМИЗАЦИЯ ---
device = 'cpu'
device_type = 'cpu'
if hasattr(torch, "xpu") and torch.xpu.is_available():
    device = 'xpu'
    device_type = 'xpu'
else:
    torch.set_num_threads(6)

console = Console()

# --- 2. НАСТРОЙКИ ---
batch_size = 64              
gradient_accumulation_steps = 2
block_size = 64              
max_iters = 20000            
eval_interval = 500          
learning_rate = 5e-4         
min_lr = 5e-5                
n_embd = 256                 
n_head = 8                   
n_layer = 6                  
dropout = 0.1        

torch.manual_seed(42)

# --- 3. АРХИТЕКТУРА ---
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.num_heads, self.head_size = num_heads, head_size
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(n_embd, dim=2)
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = nn.Sequential(nn.Linear(n_embd, 4*n_embd), nn.GELU(), nn.Linear(4*n_embd, n_embd), nn.Dropout(dropout))
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
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

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_emb(idx) + self.pos_emb(torch.arange(T, device=device))
        x = self.blocks(x)
        logits = self.lm_head(self.ln_f(x))
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1)) if targets is not None else None
        return logits, loss

# --- 4. ПОДГОТОВКА ---
def train():
    console.print(Panel("[bold cyan]Запуск процесса обучения Pythagoras v2.0[/]", expand=False))
    
    if not os.path.exists('input_math.txt'):
        console.print("[bold red]ОШИБКА: Сначала запустите prep_math.py![/]")
        return

    with open('input_math.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }

    with open('math_vocab.pkl', 'wb') as f:
        pickle.dump({'stoi': stoi, 'itos': itos, 'vocab_size': vocab_size}, f)

    encode = lambda s:[stoi.get(c, 0) for c in s]
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data))
    train_data, val_data = data[:n], data[n:]

    def get_batch(split):
        ds = train_data if split == 'train' else val_data
        ix = torch.randint(len(ds) - block_size, (batch_size,))
        x = torch.stack([ds[i:i+block_size] for i in ix]).pin_memory()
        y = torch.stack([ds[i+1:i+block_size+1] for i in ix]).pin_memory()
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    model = SimpleLLM(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    params_m = sum(p.numel() for p in model.parameters())/1e6
    console.print(f"[bold yellow]Масса мозга:[/] [bold green]{params_m:.2f}M параметров[/]")
    console.print(f"[bold yellow]Устройство:[/] [bold green]{device.upper()}[/]\n")

    def get_lr(it):
        if it < 200: return learning_rate * it / 200
        if it > max_iters: return min_lr
        decay_ratio = (it - 200) / (max_iters - 200)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)

    # --- PROGRESS UI ---
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console
    )

    stats_table = Table.grid(expand=True)
    stats_table.add_column(style="cyan")
    stats_table.add_column(justify="right", style="green")

    with Live(Panel(stats_table, title="Статистика обучения", border_style="blue"), console=console, refresh_per_second=4) as live:
        train_task = progress.add_task("[cyan]Обучение...", total=max_iters)
        start_time = time.time()
        
        last_loss = 0.0
        
        for iter in range(max_iters):
            lr = get_lr(iter)
            for param_group in optimizer.param_groups: param_group['lr'] = lr

            if iter % eval_interval == 0 or iter == max_iters - 1:
                model.eval()
                with torch.no_grad():
                    x_val, y_val = get_batch('val')
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        _, loss_val = model(x_val, y_val)
                    last_loss = loss_val.item()
                model.train()

            # UI Update
            if iter % 10 == 0:
                stats_table = Table.grid(expand=True)
                stats_table.add_row("Итерация:", f"{iter}/{max_iters}")
                stats_table.add_row("Ошибка (val):", f"{last_loss:.4f}")
                stats_table.add_row("Learning Rate:", f"{lr:.2e}")
                stats_table.add_row("Прогресс:", progress)
                live.update(Panel(stats_table, title="Pythagoras v2.0 Training", border_style="blue"))

            # Training step
            optimizer.zero_grad(set_to_none=True)
            for _ in range(gradient_accumulation_steps):
                xb, yb = get_batch('train')
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    _, loss = model(xb, yb)
                    loss = loss / gradient_accumulation_steps
                loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            progress.update(train_task, advance=1)

    torch.save(model.state_dict(), 'math_model_weights.pth')
    console.print(f"\n[bold green]✅ Обучение завершено![/] Веса сохранены в [italic]math_model_weights.pth[/]")

if __name__ == "__main__":
    train()