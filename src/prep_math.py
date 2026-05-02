import random
# --- 1.1 Графическое изображение ---
RESET  = "\033[0m"
RED    = "\033[31m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
BLUE   = "\033[34m"
PURPLE = "\033[35m"
CYAN   = "\033[36m"
WHITE  = "\033[37m"
LIME       = "\033[38;5;118m"  
HOT_PINK   = "\033[38;5;205m"  
NEON_BLUE  = "\033[38;5;51m"   
VIOLET     = "\033[38;5;177m"  
ORANGE     = "\033[38;5;208m"  
GOLD       = "\033[38;5;220m"  
TOMATO     = "\033[38;5;202m"  
SKY        = "\033[38;5;117m"  
LAVENDER   = "\033[38;5;183m"  
SAND       = "\033[38;5;230m"  
GREY       = "\033[38;5;245m"  

print(SKY + r"""
  _____        _   _                                   
 |  __ \      | | | |                                  
 | |__) |   _ | |_| |__   __ _  __ _  ___  _ __ __ _ ___ 
 |  ___/ | | || __| '_ \ / _` |/ _` |/ _ \| '__/ _` / __|
 | |   | |_| || |_| | | | (_| | (_| | (_) | | | (_| \__ \
 |_|    \__, | \__|_| |_|\__,_|\__, |\___/|_|  \__,_|___/
         __/ |                  __/ |                    
        |___/                  |___/   [ v2.0 ]
""" + RESET)
def generate_balanced_math():
    print("⚖️ Генерируем идеально сбалансированный датасет...")
    examples = []
    
    # Конфигурация: сколько примеров каждой сложности нам нужно
    # Мы делим по количеству знаков в ОПЕРАНДАХ (слагаемых)
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
            
            if op == '+':
                res = a + b
            else:
                # Гарантируем отсутствие отрицательных для чистоты обучения
                if a < b: a, b = b, a
                res = a - b
            
            examples.append(f"{a}{op}{b}={res}\n")

    # Добавляем "Критическую массу" особых случаев (10% от общего числа)
    # Это учит ИИ свойствам нуля и одинаковых чисел
    for _ in range(50000):
        a = random.randint(0, 999)
        case = random.choice([
            (a, 0, '+'),  # x + 0
            (a, 0, '-'),  # x - 0
            (a, a, '-'),  # x - x (база нуля)
            (0, a, '+'),  # 0 + x
        ])
        a_val, b_val, op = case
        res = a_val + b_val if op == '+' else a_val - b_val
        examples.append(f"{a_val}{op}{b_val}={res}\n")

    # ВАЖНО: Перемешиваем всё! 
    # Если сначала пойдут только легкие, а потом только сложные, ИИ все забудет.
    random.shuffle(examples)

    import os
    os.makedirs('data', exist_ok=True)
    with open('data/input_math.txt', 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(ex)

    print(f"✅ Создано {len(examples)} сбалансированных примеров в 'data/input_math.txt'")

if __name__ == "__main__":
    generate_balanced_math()