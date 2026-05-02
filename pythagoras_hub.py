import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
import time
import os
import math
import datetime
import psutil
import random
import csv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.align import Align
from rich.status import Status
from rich.columns import Columns
from rich.text import Text
from rich.box import ROUNDED, DOUBLE_EDGE
try:
    import matplotlib
    matplotlib.use('Agg')  # Не-интерактивный бэкенд — рисует в файл без окна
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# --- 1. ГЛОБАЛЬНЫЕ НАСТРОЙКИ ---
console = Console()
device = 'cpu'
device_type = 'cpu'
if hasattr(torch, "xpu") and torch.xpu.is_available():
    device = 'xpu'
    device_type = 'xpu'
else:
    torch.set_num_threads(6)

# Гиперпараметры (v2.0)
n_embd = 256
n_head = 8
n_layer = 6
block_size = 64
dropout = 0.1

# --- 2. АРХИТЕКТУРА МОДЕЛИ ---
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

# --- 3. ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ГЕНЕРАЦИИ ОТЧЁТОВ ---

REPORTS_DIR = "reports"

def generate_validation_report(results: list, patterns: dict, accuracy: float, correct_count: int):
    """Генерирует CSV-таблицу и PNG-диаграмму паттернов по итогам валидации."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # --- 3a. CSV: Полная таблица 1000 примеров ---
    csv_filename = os.path.join(REPORTS_DIR, f"validation_results_{timestamp}.csv")
    try:
        with open(csv_filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
            # utf-8-sig гарантирует корректное открытие в Excel (BOM)
            writer = csv.writer(csvfile, delimiter=';')
            # Заголовок
            writer.writerow(["№", "Пример", "Ожидаемый ответ", "Ответ нейросети", "Статус", "Паттерн сложности"])
            for i, r in enumerate(results, 1):
                status_text = "✓ Верно" if r['ok'] else "✗ Ошибка"
                writer.writerow([i, r['prompt'], r['target'], r['pred'], status_text, r['pattern']])
            # Пустая строка-разделительЫ
            writer.writerow([])
            # Итоговая строка
            writer.writerow(["ИТОГО", f"{len(results)} тестов", "", "", f"Точность: {accuracy:.1f}%", f"{correct_count} верных / {len(results) - correct_count} ошибок"])
        csv_saved = True
    except Exception as e:
        csv_saved = False
        csv_error = str(e)

    # --- 3b. PNG: Диаграмма точности по паттернам ---
    chart_filename = os.path.join(REPORTS_DIR, f"validation_patterns_{timestamp}.png")
    chart_saved = False
    if MATPLOTLIB_AVAILABLE:
        try:
            sorted_patterns = sorted(patterns.keys())
            pattern_labels = sorted_patterns
            correct_vals = [patterns[k][0] for k in sorted_patterns]
            error_vals   = [patterns[k][1] - patterns[k][0] for k in sorted_patterns]
            accuracy_pct = [patterns[k][0] / patterns[k][1] * 100 for k in sorted_patterns]

            fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor='#1a1a2e')
            fig.suptitle(
                f'Pythagoras 1.0 — Отчёт о валидации  •  {timestamp}',
                fontsize=14, color='#e0e0e0', fontweight='bold', y=1.01
            )

            # ── Левый график: Стопчатая диаграмма «верно / ошибка» ──────────
            ax1 = axes[0]
            ax1.set_facecolor('#16213e')
            x = range(len(sorted_patterns))
            bar_w = 0.6
            bars_ok  = ax1.bar(x, correct_vals, bar_w, label='Верно',  color='#00c897', zorder=3)
            bars_err = ax1.bar(x, error_vals,   bar_w, label='Ошибка', color='#ff4d6d',
                               bottom=correct_vals, zorder=3)

            # Подписи % над каждым столбцом
            for xi, (cv, ev, acc) in enumerate(zip(correct_vals, error_vals, accuracy_pct)):
                total = cv + ev
                ax1.text(xi, total + 0.5, f'{acc:.1f}%', ha='center', va='bottom',
                         fontsize=9, color='#e0e0e0', fontweight='bold')

            ax1.set_xticks(list(x))
            ax1.set_xticklabels(pattern_labels, rotation=35, ha='right',
                                fontsize=9, color='#c0c0c0')
            ax1.set_yticks(range(0, max(correct_vals[i] + error_vals[i] for i in range(len(sorted_patterns))) + 10, 10))
            ax1.set_yticklabels(ax1.get_yticks(), color='#c0c0c0')
            ax1.set_ylabel('Количество примеров', color='#c0c0c0')
            ax1.set_title('Распределение верных / ошибочных ответов', color='#e0e0e0', pad=10)
            ax1.tick_params(colors='#c0c0c0')
            ax1.spines[:].set_color('#334455')
            ax1.grid(axis='y', color='#334455', linestyle='--', alpha=0.6, zorder=0)
            ax1.legend(handles=[
                mpatches.Patch(color='#00c897', label='Верно'),
                mpatches.Patch(color='#ff4d6d', label='Ошибка')
            ], facecolor='#1a1a2e', labelcolor='#e0e0e0', framealpha=0.8)

            # ── Правый график: Горизонтальный bar с % точности ──────────────
            ax2 = axes[1]
            ax2.set_facecolor('#16213e')

            def acc_color(pct):
                if pct >= 90: return '#00c897'
                elif pct >= 70: return '#f9c74f'
                else: return '#ff4d6d'

            colors = [acc_color(a) for a in accuracy_pct]
            y_pos = range(len(sorted_patterns))
            bars_h = ax2.barh(list(y_pos), accuracy_pct, 0.55, color=colors, zorder=3)

            # Подписи значений внутри/снаружи полос
            for bar, acc in zip(bars_h, accuracy_pct):
                xval = bar.get_width()
                ax2.text(min(xval + 1.5, 98), bar.get_y() + bar.get_height() / 2,
                         f'{acc:.1f}%', va='center', fontsize=9,
                         color='#e0e0e0', fontweight='bold')

            ax2.set_yticks(list(y_pos))
            ax2.set_yticklabels(sorted_patterns, fontsize=9, color='#c0c0c0')
            ax2.set_xlim(0, 110)
            ax2.set_xlabel('Точность (%)', color='#c0c0c0')
            ax2.set_title('Процент точности по паттернам', color='#e0e0e0', pad=10)
            ax2.tick_params(colors='#c0c0c0')
            ax2.spines[:].set_color('#334455')
            ax2.grid(axis='x', color='#334455', linestyle='--', alpha=0.6, zorder=0)
            ax2.axvline(x=90, color='#00c897', linestyle=':', linewidth=1.2, label='Порог 90%')
            ax2.axvline(x=70, color='#f9c74f', linestyle=':', linewidth=1.2, label='Порог 70%')
            ax2.legend(facecolor='#1a1a2e', labelcolor='#e0e0e0', framealpha=0.8, fontsize=8)

            # Общая надпись об итоге
            total_color = acc_color(accuracy)
            fig.text(0.5, -0.02,
                     f'Общая точность: {accuracy:.1f}%  •  Верно: {correct_count} / {len(results)}  •  Ошибок: {len(results) - correct_count}',
                     ha='center', fontsize=11, color=total_color, fontweight='bold')

            plt.tight_layout()
            plt.savefig(chart_filename, dpi=150, bbox_inches='tight',
                        facecolor='#1a1a2e', edgecolor='none')
            plt.close(fig)
            chart_saved = True
        except Exception as e:
            chart_saved = False
            chart_error = str(e)
    
    # --- Красивый вывод сохранённых путей ---
    lines = ["[bold white]📂 Отчёты сохранены в папку [cyan]reports/[/cyan][/bold white]\n"]
    if csv_saved:
        lines.append(f"  [green]✓ CSV таблица:[/]  [white]{csv_filename}[/]")
    else:
        lines.append(f"  [red]✗ CSV ошибка:[/] [dim]{csv_error}[/]")

    if not MATPLOTLIB_AVAILABLE:
        lines.append("  [yellow]⚠ График пропущен:[/] [dim]установите matplotlib (pip install matplotlib)[/]")
    elif chart_saved:
        lines.append(f"  [green]✓ Диаграмма PNG:[/]  [white]{chart_filename}[/]")
    else:
        lines.append(f"  [red]✗ График ошибка:[/] [dim]{chart_error}[/]")

    console.print(Panel("\n".join(lines), title="[bold cyan]📊 Инфографика[/]",
                        border_style="cyan", box=ROUNDED, padding=(0, 2)))


# --- 4. ФУНКЦИИ ИНТЕРФЕЙСА ---

def print_banner():
    banner = r"""
 [bold white]
 ██████╗ ██╗   ██╗████████╗██╗  ██╗ █████╗  ██████╗  ██████╗ ██████╗  █████╗ ███████╗
 ██╔══██╗╚██╗ ██╔╝╚══██╔══╝██║  ██║██╔══██╗██╔════╝ ██╔═══██╗██╔══██╗██╔══██╗██╔════╝
 [/][bold blue]
 ██████╔╝ ╚████╔╝    ██║   ███████║███████║██║  ███╗██║   ██║██████╔╝███████║███████╗
 ██╔═══╝   ╚██╔╝     ██║   ██╔══██║██╔══██║██║   ██║██║   ██║██╔══██╗██╔══██║╚════██║
 [/][bold red]
 ██║        ██║      ██║   ██║  ██║██║  ██║╚██████╔╝╚██████╔╝██║  ██║██║  ██║███████║
 ╚═╝        ╚═╝      ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝
 [/]
    """
    
    math_symbols = "∫ ∑ √ π Ω ∞ ∆ ≈ ≡ × ÷"
    
    console.print(Panel(
        Align.center(banner, vertical="middle"), 
        style="bright_blue", 
        box=DOUBLE_EDGE,
        title=f"[bold white] {math_symbols} [/]", 
        subtitle=f"[bold white] {math_symbols} [/]"
    ))

def mode_chat():
    console.clear()
    console.print(Panel("[bold blue]🗨️ РЕЖИМ ИНТЕРФЕЙСА С ПИФАГОРОМ[/]", border_style="blue", box=ROUNDED))
    
    if not os.path.exists('weights/math_model_weights.pth'):
        console.print(Panel("[bold red]❌ ОШИБКА: Веса модели не найдены. Сначала проведите обучение![/]", border_style="red"))
        return

    with Status("[bold green]🧠 Инициализация нейронов...", console=console):
        with open('weights/math_vocab.pkl', 'rb') as f:
            vocab_data = pickle.load(f)
        stoi, itos, vocab_size = vocab_data['stoi'], vocab_data['itos'], vocab_data['vocab_size']
        
        model = SimpleLLM(vocab_size).to(device)
        model.load_state_dict(torch.load('weights/math_model_weights.pth', map_location=device, weights_only=True))
        model.eval()

    encode = lambda s: [stoi.get(c, 0) for c in s]
    decode = lambda l: ''.join([itos.get(i, '?') for i in l])

    console.print("[dim]Подсказка: Введите 'back' для выхода. Например: 25+75=[/]")

    while True:
        prompt = Prompt.ask("\n[bold cyan]?[/] [white]Математический запрос[/]")
        if prompt.lower() == 'back': break
        if not prompt: continue
        if '=' not in prompt: prompt += '='

        context = torch.tensor((encode(prompt),), dtype=torch.long, device=device)
        generated_idx = context.tolist()[0]
        
        with console.status("[italic yellow]Вычисляю квантовую вероятность...[/]", spinner="dots9"):
            for _ in range(15):
                idx_cond = torch.tensor((generated_idx[-block_size:],), dtype=torch.long, device=device)
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, _ = model(idx_cond)
                    idx_next = torch.argmax(logits[0, -1, :]).item()
                    generated_idx.append(idx_next)
                    if itos.get(idx_next) == '\n': break
                    
        result = decode(generated_idx).replace('\n', '').strip()
        
        # Красивый вывод ответа
        res_text = Text()
        res_text.append("Пифагор: ", style="bold green")
        res_text.append(result, style="bold yellow")
        
        console.print(Panel(Align.center(res_text), border_style="bright_green", box=ROUNDED, expand=False))
        
        # Логирование
        os.makedirs("logs", exist_ok=True)
        with open("logs/math_chat_history.txt", "a", encoding="utf-8") as f:
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            f.write(f"[{ts}] Запрос: {prompt} | Ответ: {result}\n")

def mode_train():
    console.clear()
    console.print(Panel("[bold magenta]🧬 РЕЖИМ КЛИНИЧЕСКОГО ОБУЧЕНИЯ[/]", border_style="magenta", box=ROUNDED))
    
    if not os.path.exists('data/input_math.txt'):
        console.print("[bold yellow]⚠️ Датасет не обнаружен![/]")
        if Prompt.ask("Сгенерировать новые данные?", choices=["y", "n"]) == "y":
            mode_dataset()
        else: return

    max_iters = IntPrompt.ask("[bold white]Количество циклов обучения[/]", default=5000)
    
    with Status("[bold magenta]📚 Загрузка знаний в память...", console=console):
        with open('data/input_math.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        chars = sorted(list(set(text)))
        vocab_size = len(chars)
        stoi = { ch:i for i,ch in enumerate(chars) }
        itos = { i:ch for i,ch in enumerate(chars) }
        os.makedirs("weights", exist_ok=True)
        with open('weights/math_vocab.pkl', 'wb') as f:
            pickle.dump({'stoi': stoi, 'itos': itos, 'vocab_size': vocab_size}, f)
        
        encode = lambda s:[stoi.get(c, 0) for c in s]
        data = torch.tensor(encode(text), dtype=torch.long)
        n = int(0.9*len(data))
        train_data, val_data = data[:n], data[n:]

    def get_batch(split):
        ds = train_data if split == 'train' else val_data
        ix = torch.randint(len(ds) - block_size, (batch_size,))
        x = torch.stack([ds[i:i+block_size] for i in ix])
        y = torch.stack([ds[i+1:i+block_size+1] for i in ix])
        return x.to(device), y.to(device)

    batch_size = 64
    model = SimpleLLM(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    
    progress = Progress(
        SpinnerColumn("pixel"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40, gradient=("magenta", "blue")),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    )

    with Live(Panel(Align.center("[bold]Подготовка нейросетевых связей...[/]"), title="Мониторинг"), console=console, refresh_per_second=10) as live:
        task = progress.add_task("[magenta]Синтез логики...", total=max_iters)
        for iter in range(max_iters):
            if iter % 200 == 0:
                model.eval()
                with torch.no_grad():
                    xb, yb = get_batch('val')
                    _, loss = model(xb, yb)
                    cur_loss = loss.item()
                model.train()
                
                # Обновление дашборда
                table = Table.grid(expand=True)
                table.add_row(f"Цикл: [bold]{iter}/{max_iters}[/]")
                table.add_row(f"Ошибка: [bold green]{cur_loss:.4f}[/]")
                table.add_row(progress)
                live.update(Panel(table, title="[bold magenta]Training Dashboard[/]", border_style="magenta"))

            optimizer.zero_grad(set_to_none=True)
            xb, yb = get_batch('train')
            _, loss = model(xb, yb)
            loss.backward()
            optimizer.step()
            progress.update(task, advance=1)

    torch.save(model.state_dict(), 'weights/math_model_weights.pth')
    console.print(Panel("[bold green]🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО![/]", border_style="green"))

def mode_history():
    console.clear()
    console.print(Panel("[bold yellow]📜 АРХИВ ВЫЧИСЛЕНИЙ[/]", border_style="yellow", box=ROUNDED))
    
    if not os.path.exists('logs/math_chat_history.txt'):
        console.print("[dim]История пуста...[/]")
    else:
        table = Table(title="Последние 20 записей", box=ROUNDED, header_style="bold yellow", expand=True)
        table.add_column("Время", style="dim", width=10)
        table.add_column("Запрос", style="cyan")
        table.add_column("Ответ Пифагора", style="bold green")

        with open('logs/math_chat_history.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()[-20:]
            for line in lines:
                if "|" in line:
                    try:
                        time_part = line.split("]")[0].replace("[", "")
                        query_part = line.split("Запрос:")[1].split("|")[0].strip()
                        answer_part = line.split("Ответ:")[1].strip()
                        table.add_row(time_part, query_part, answer_part)
                    except: continue
        
        console.print(table)
    
    Prompt.ask("\n[bold white]Нажмите Enter, чтобы вернуться в меню[/]")

def mode_debug():
    console.clear()
    console.print(Panel("[bold red]🛠️ ЦЕНТР ОТЛАДКИ И ДИАГНОСТИКИ[/]", border_style="red", box=ROUNDED))
    
    while True:
        options = [
            Panel("[bold white]1. Мониторинг ресурсов[/]\n[dim]CPU, RAM, VRAM[/]", border_style="white", box=ROUNDED),
            Panel("[bold white]2. Сканирование тензоров[/]\n[dim]Архитектура слоев[/]", border_style="white", box=ROUNDED),
            Panel("[bold white]3. Тест валидности (Arithm)[/]\n[dim]Проверка 100 примеров[/]", border_style="white", box=ROUNDED),
            Panel("[bold red]4. Назад[/]\n[dim]В главное меню[/]", border_style="red", box=ROUNDED)
        ]
        console.print(Columns(options, equal=True, expand=True))
        
        choice = Prompt.ask("\n[bold white]Выберите диагностику[/]", choices=["1", "2", "3", "4"])
        
        if choice == "1":
            cpu_usage = psutil.cpu_percent(interval=1)
            ram = psutil.virtual_memory()
            vram_text = "N/A"
            if device == 'xpu': vram_text = "Intel Arc Active"
            elif device == 'cuda': vram_text = f"{torch.cuda.memory_allocated() / 1024**2:.1f} MB"
            
            table = Table(title="Системные ресурсы", box=ROUNDED)
            table.add_column("Параметр", style="cyan")
            table.add_column("Значение", style="bold yellow")
            table.add_row("Загрузка CPU", f"{cpu_usage}%")
            table.add_row("Использование RAM", f"{ram.percent}% ({ram.used / 1024**3:.1f} GB)")
            table.add_row("Видеопамять (VRAM)", vram_text)
            table.add_row("Потоков Torch", str(torch.get_num_threads()))
            console.print(table)
            Prompt.ask("\nНажмите Enter...")
            
        elif choice == "2":
            if not os.path.exists('weights/math_model_weights.pth'):
                console.print("[red]Модель не загружена![/]")
                continue
            
            with open('weights/math_vocab.pkl', 'rb') as f: v_data = pickle.load(f)
            m = SimpleLLM(v_data['vocab_size']).to(device)
            m.load_state_dict(torch.load('weights/math_model_weights.pth', map_location=device, weights_only=True))
            
            table = Table(title="Архитектура тензоров (Pythagoras v2.2)", box=ROUNDED)
            table.add_column("Слой", style="cyan")
            table.add_column("Форма (Shape)", style="green")
            table.add_column("Параметры", style="yellow")
            
            for name, param in m.named_parameters():
                table.add_row(name, str(list(param.shape)), str(param.numel()))
            
            console.print(table)
            console.print(f"\n[bold white]Всего параметров:[/] [bold green]{sum(p.numel() for p in m.parameters()):,}[/]")
            Prompt.ask("\nНажмите Enter...")

        elif choice == "3":
            if not os.path.exists('weights/math_model_weights.pth'):
                console.print("[red]Ошибка: Обучите модель![/]")
                continue
                
            with Status("[bold yellow]Запуск расширенной валидации (1000 примеров)...", console=console):
                with open('weights/math_vocab.pkl', 'rb') as f: v_data = pickle.load(f)
                stoi, itos = v_data['stoi'], v_data['itos']
                m = SimpleLLM(v_data['vocab_size']).to(device)
                m.load_state_dict(torch.load('weights/math_model_weights.pth', map_location=device, weights_only=True))
                m.eval()
                
                encode = lambda s: [stoi.get(c, 0) for c in s]
                decode = lambda l: ''.join([itos.get(i, '?') for i in l])
                
                # Генерируем сбалансированные паттерны (15 комбинаций)
                combos = []
                for la in [1, 2, 3]:
                    for lb in [1, 2, 3]:
                        combos.append((la, '+', lb))
                        if la >= lb: combos.append((la, '-', lb))
                
                results = []
                patterns = {}
                total_tests = 1000
                tests_per_combo = total_tests // len(combos)

                for la, op, lb in combos:
                    for _ in range(tests_per_combo):
                        min_a, max_a = (10**(la-1) if la>1 else 0), (10**la - 1)
                        min_b, max_b = (10**(lb-1) if lb>1 else 0), (10**lb - 1)
                        a, b = random.randint(min_a, max_a), random.randint(min_b, max_b)
                        if op == '-' and a < b: a, b = b, a
                        
                        target = a + b if op == '+' else a - b
                        prompt = f"{a}{op}{b}="
                        pattern_key = f"{la}d {op} {lb}d"
                        
                        if pattern_key not in patterns: patterns[pattern_key] = [0, 0]
                        patterns[pattern_key][1] += 1

                        context = torch.tensor((encode(prompt),), dtype=torch.long, device=device)
                        gen_idx = context.tolist()[0]
                        for _ in range(10):
                            idx_c = torch.tensor((gen_idx[-block_size:],), dtype=torch.long, device=device)
                            with torch.no_grad():
                                logits, _ = m(idx_c)
                                nxt = torch.argmax(logits[0, -1, :]).item()
                                gen_idx.append(nxt)
                                if itos.get(nxt) == '\n': break
                        
                        pred_str = decode(gen_idx).split('=')[1].strip()
                        try:
                            is_ok = int(pred_str) == target
                            if is_ok: 
                                patterns[pattern_key][0] += 1
                            results.append({'prompt': prompt, 'target': target, 'pred': pred_str, 'ok': is_ok, 'pattern': pattern_key})
                        except:
                            results.append({'prompt': prompt, 'target': target, 'pred': pred_str, 'ok': False, 'pattern': pattern_key})

            # --- ОТЧЕТ ПО ВАЛИДАЦИИ ---
            correct_count = sum(1 for r in results if r['ok'])
            accuracy = (correct_count / len(results)) * 100
            
            score_color = "green" if accuracy > 90 else "yellow" if accuracy > 70 else "red"
            console.print(Panel(Align.center(f"[bold {score_color}]ОБЩАЯ ТОЧНОСТЬ: {accuracy:.1f}% ({correct_count}/{len(results)})[/]"), title="Итоговый отчет"))
            
            # Таблица паттернов
            p_table = Table(title="Анализ паттернов сложности", box=ROUNDED)
            p_table.add_column("Тип примера", style="cyan")
            p_table.add_column("Точность", justify="right")
            p_table.add_column("Статус", justify="center")

            for p_key in sorted(patterns.keys()):
                succ, tot = patterns[p_key]
                p_acc = (succ / tot) * 100
                p_color = "green" if p_acc > 90 else "yellow" if p_acc > 50 else "red"
                status_icon = "✅" if p_acc == 100 else "⚠️" if p_acc > 50 else "❌"
                p_table.add_row(p_key, f"[{p_color}]{p_acc:.1f}% ({succ}/{tot})[/]", status_icon)
            
            console.print(p_table)

            # Детальный разбор ошибок
            errors = [r for r in results if not r['ok']]
            if errors:
                e_table = Table(title="Топ ошибок для отладки", box=ROUNDED)
                e_table.add_column("Пример", style="white")
                e_table.add_column("Ожидалось", style="green")
                e_table.add_column("ИИ", style="red")
                e_table.add_column("Паттерн", style="dim")
                for e in errors[:10]:
                    e_table.add_row(e['prompt'], str(e['target']), e['pred'], e['pattern'])
                console.print(e_table)
                
                # Вывод самого слабого звена
                weakest = min(patterns.items(), key=lambda x: x[1][0]/x[1][1])
                console.print(Panel(f"[bold red]Критическая уязвимость:[/] Модель хуже всего справляется с паттерном [bold cyan]{weakest[0]}[/] ({weakest[1][0]/weakest[1][1]*100:.1f}%)", border_style="red"))
            else:
                console.print("[bold green]✨ Идеальный результат! Все паттерны освоены.[/]")

            # --- Генерация инфографики ---
            with console.status("[bold cyan]📊 Генерирую инфографику и экспортирую отчёты...[/]", spinner="dots"):
                generate_validation_report(results, patterns, accuracy, correct_count)

            Prompt.ask("\nНажмите Enter...")
            
def mode_dataset():
    console.clear()
    console.print(Panel("[bold cyan]⚖️ ГЕНЕРАЦИЯ МАТЕМАТИЧЕСКОГО ДАТАСЕТА[/]", border_style="cyan", box=ROUNDED))
    
    with console.status("[bold cyan]Синтезируем идеально сбалансированные примеры...", spinner="earth"):
        examples = []
        # Конфигурация: сколько примеров каждой сложности нам нужно
        configs = [
            (1, 9, 100000),      # 100к примеров типа 1+1, 5-3 (односложные)
            (10, 99, 150000),    # 150к примеров типа 15+20, 80-40 (двузначные)
            (100, 999, 250000),  # 250к примеров типа 120+450 (трехзначные)
            (1, 999, 50000),     # 50к смешанных: 1+999, 10+500
        ]

        for min_v, max_v, count in configs:
            for _ in range(count):
                a = random.randint(min_v, max_v)
                b = random.randint(min_v, max_v)
                op = random.choice(['+', '-'])
                if op == '+': res = a + b
                else:
                    if a < b: a, b = b, a
                    res = a - b
                examples.append(f"{a}{op}{b}={res}\n")

        # Критическая масса особых случаев (50к)
        for _ in range(50000):
            a = random.randint(0, 999)
            case = random.choice([(a, 0, '+'), (a, 0, '-'), (a, a, '-'), (0, a, '+')])
            a_val, b_val, op = case
            res = a_val + b_val if op == '+' else a_val - b_val
            examples.append(f"{a_val}{op}{b_val}={res}\n")

        random.shuffle(examples)
        os.makedirs('data', exist_ok=True)
        with open('data/input_math.txt', 'w', encoding='utf-8') as f:
            for ex in examples:
                f.write(ex)

    console.print(Panel(f"[bold green]✅ Создано {len(examples)} сбалансированных примеров!\n[white]Файл: [cyan]data/input_math.txt[/]", border_style="green"))
    Prompt.ask("\nНажмите Enter, чтобы вернуться в меню")

# --- 4. ГЛАВНОЕ МЕНЮ ---

def main():
    while True:
        console.clear()
        print_banner()
        
        options = [
            Panel("[bold cyan]1. ВХОД В ЧАТ[/]\n[dim]Диалог с ИИ[/]", border_style="cyan", box=ROUNDED),
            Panel("[bold magenta]2. ОБУЧЕНИЕ[/]\n[dim]Тренировка[/]", border_style="magenta", box=ROUNDED),
            Panel("[bold blue]3. ДАТАСЕТ[/]\n[dim]Генерация[/]", border_style="blue", box=ROUNDED),
            Panel("[bold yellow]4. ИСТОРИЯ[/]\n[dim]Архив логов[/]", border_style="yellow", box=ROUNDED),
            Panel("[bold red]5. ОТЛАДКА[/]\n[dim]Диагностика[/]", border_style="red", box=ROUNDED),
            Panel("[bold white]6. ВЫХОД[/]\n[dim]Закрыть[/]", border_style="white", box=ROUNDED)
        ]
        
        console.print(Columns(options, equal=True, expand=True))
        
        choice = Prompt.ask("\n[bold white]Выберите сектор[/]", choices=["1", "2", "3", "4", "5", "6"])
        
        if choice == "1": mode_chat()
        elif choice == "2": mode_train()
        elif choice == "3": mode_dataset()
        elif choice == "4": mode_history()
        elif choice == "5": mode_debug()
        else:
            console.print("[italic red]Система отключена.[/]")
            break

if __name__ == "__main__":
    main()