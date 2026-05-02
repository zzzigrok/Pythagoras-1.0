import { useState, useEffect, useRef } from 'react';

// --- Пользовательские стили для специфичных анимаций ---
const CustomStyles = () => (
  <style>
    {`
      @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(5deg); }
      }
      @keyframes float-reverse {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(20px) rotate(-5deg); }
      }
      @keyframes blob-breathe {
        0%, 100% { transform: scale(1); opacity: 0.8; }
        50% { transform: scale(1.1); opacity: 1; }
      }
      .animate-float { animation: float 6s ease-in-out infinite; }
      .animate-float-delayed { animation: float-reverse 8s ease-in-out infinite 2s; }
      .animate-float-slow { animation: float 10s ease-in-out infinite 1s; }
      .animate-blob { animation: blob-breathe 8s infinite alternate ease-in-out; }
    `}
  </style>
);

// --- Компонент для плавного появления при скролле ---
const RevealOnScroll = ({ children, className = '', delay = 0, threshold = 0.1, direction = 'up' }: any) => {
  const [isVisible, setIsVisible] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          observer.unobserve(entry.target);
        }
      },
      { threshold, rootMargin: '0px 0px -50px 0px' }
    );
    if (ref.current) observer.observe(ref.current);
    return () => observer.disconnect();
  }, [threshold]);

  const getTransform = () => {
    if (isVisible) return 'translate-y-0 translate-x-0 scale-100';
    switch (direction) {
      case 'up': return 'translate-y-12 scale-95';
      case 'left': return '-translate-x-12';
      case 'right': return 'translate-x-12';
      default: return 'translate-y-12';
    }
  };

  return (
    <div
      ref={ref}
      style={{ transitionDelay: `${delay}ms` }}
      className={`transition-all duration-1000 ease-out ${isVisible ? 'opacity-100' : 'opacity-0'} ${getTransform()} ${className}`}
    >
      {children}
    </div>
  );
};

// --- Фоновые плавающие математические символы ---
const FloatingMath = () => {
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none z-0">
      <div className="absolute top-[20%] left-[10%] text-blue-500/10 text-6xl animate-float font-serif">∑</div>
      <div className="absolute top-[60%] right-[15%] text-purple-500/10 text-8xl animate-float-delayed font-serif">∫</div>
      <div className="absolute bottom-[20%] left-[20%] text-teal-500/10 text-5xl animate-float-slow font-serif">π</div>
      <div className="absolute top-[30%] right-[25%] text-blue-400/10 text-4xl animate-float font-serif">∆</div>
      <div className="absolute bottom-[40%] right-[35%] text-purple-400/10 text-7xl animate-float-delayed font-serif">∞</div>
    </div>
  );
};

// --- Главный фоновый грид ---
const GridBackground = () => (
  <svg className="absolute inset-0 w-full h-full opacity-[0.15] pointer-events-none" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <pattern id="grid-pattern" width="40" height="40" patternUnits="userSpaceOnUse">
        <path d="M 40 0 L 0 0 0 40" fill="none" stroke="currentColor" strokeWidth="0.5" className="text-blue-400" />
      </pattern>
    </defs>
    <rect width="100%" height="100%" fill="url(#grid-pattern)" />
  </svg>
);

// --- Анимированный треугольник Пифагора ---
const AnimatedTriangle = () => {
  return (
    <div className="relative w-full max-w-md mx-auto mb-12 lg:mb-0 animate-float drop-shadow-[0_0_30px_rgba(59,130,246,0.3)]">
      <svg viewBox="0 0 100 100" className="w-full h-full">
        <defs>
          <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="2" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
        </defs>

        {/* Правильный прямоугольный треугольник Пифагора (пропорции 3-4-5) */}
        <polygon 
          points="55,25 85,65 55,65" 
          fill="none" 
          stroke="currentColor" 
          strokeWidth="1.5"
          className="text-blue-400"
          filter="url(#glow)"
        >
          <animate 
            attributeName="stroke-dasharray" 
            values="0,150; 150,0" 
            dur="6s" 
            repeatCount="indefinite" 
            calcMode="spline"
            keySplines="0.4 0 0.2 1"
          />
        </polygon>
        
        {/* Квадраты Пифагора (a^2 + b^2 = c^2) с эффектами наведения */}
        <rect x="15" y="25" width="40" height="40" fill="currentColor" className="text-purple-500/20 stroke-purple-500/60 stroke-1 cursor-pointer transition-all hover:text-purple-500/40" />
        <rect x="55" y="65" width="30" height="30" fill="currentColor" className="text-teal-500/20 stroke-teal-500/60 stroke-1 cursor-pointer transition-all hover:text-teal-500/40" />
        <polygon points="55,25 85,65 45,95 15,55" fill="currentColor" className="text-blue-500/20 stroke-blue-500/60 stroke-1 cursor-pointer transition-all hover:text-blue-500/40" />
        
        {/* Анимация подсветки периметра квадратов */}
        <rect x="15" y="25" width="40" height="40" fill="none" className="stroke-purple-400 stroke-[0.5] opacity-50">
           <animate attributeName="stroke-dasharray" values="0,160; 160,0" dur="8s" repeatCount="indefinite" />
        </rect>
        <rect x="55" y="65" width="30" height="30" fill="none" className="stroke-teal-400 stroke-[0.5] opacity-50">
           <animate attributeName="stroke-dasharray" values="0,120; 120,0" dur="6s" repeatCount="indefinite" begin="1s" />
        </rect>
        <polygon points="55,25 85,65 45,95 15,55" fill="none" className="stroke-blue-400 stroke-[0.5] opacity-50">
           <animate attributeName="stroke-dasharray" values="0,200; 200,0" dur="10s" repeatCount="indefinite" begin="2s" />
        </polygon>

        {/* Узлы (имитация слоев нейросети) */}
        <circle cx="55" cy="25" r="2.5" className="text-white fill-current animate-pulse shadow-lg" filter="url(#glow)">
           <animate attributeName="r" values="2;3.5;2" dur="2s" repeatCount="indefinite" />
        </circle>
        <circle cx="85" cy="65" r="2.5" className="text-white fill-current animate-pulse shadow-lg" filter="url(#glow)">
            <animate attributeName="r" values="2;3.5;2" dur="2.5s" repeatCount="indefinite" begin="0.5s" />
        </circle>
        <circle cx="55" cy="65" r="2.5" className="text-white fill-current animate-pulse shadow-lg" filter="url(#glow)">
            <animate attributeName="r" values="2;3.5;2" dur="3s" repeatCount="indefinite" begin="1s" />
        </circle>

        {/* Формулы в центрах квадратов */}
        <text x="35" y="48" fill="currentColor" fontSize="8" textAnchor="middle" className="text-purple-200 font-serif font-bold opacity-90">a²</text>
        <text x="70" y="83" fill="currentColor" fontSize="8" textAnchor="middle" className="text-teal-200 font-serif font-bold opacity-90">b²</text>
        <text x="50" y="63" fill="currentColor" fontSize="8" textAnchor="middle" className="text-blue-200 font-serif font-bold opacity-90" transform="rotate(-53.13, 50, 60)">c²</text>

        {/* Внутренние связи */}
        <line x1="35" y1="45" x2="55" y2="45" stroke="currentColor" strokeWidth="0.5" strokeDasharray="1,2" className="text-white/60" />
        <line x1="70" y1="80" x2="70" y2="65" stroke="currentColor" strokeWidth="0.5" strokeDasharray="1,2" className="text-white/60" />
        <line x1="50" y1="60" x2="70" y2="45" stroke="currentColor" strokeWidth="0.5" strokeDasharray="1,2" className="text-white/60" />
      </svg>
    </div>
  );
};

const FeatureIcon = ({ type }: { type: string }) => {
  switch (type) {
    case 'brain':
      return (
        <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
        </svg>
      );
    case 'code':
      return (
        <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
        </svg>
      );
    case 'lightning':
      return (
        <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
      );
    default:
      return null;
  }
};

const Section = ({ title, children, className = '' }: any) => (
  <section className={`py-24 px-6 max-w-6xl mx-auto relative ${className}`}>
    <RevealOnScroll>
      <h2 className="text-3xl md:text-5xl font-extrabold mb-16 text-center bg-clip-text text-transparent bg-gradient-to-r from-blue-400 via-purple-400 to-teal-400 drop-shadow-sm">
        {title}
      </h2>
    </RevealOnScroll>
    {children}
  </section>
);

export default function App() {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50);
    };
    window.addEventListener('scroll', handleScroll);
    // Начальная проверка
    handleScroll();
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <div className="min-h-screen bg-[#020617] text-slate-200 font-sans selection:bg-blue-500/30 overflow-x-hidden relative">
      <CustomStyles />
      
      {/* Декоративные фоновые элементы для эффекта стекла (Анимированные) */}
      <div className="fixed top-0 left-0 w-full h-full overflow-hidden pointer-events-none z-0">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] rounded-full bg-blue-600/20 blur-[120px] animate-blob"></div>
        <div className="absolute bottom-[-10%] right-[-10%] w-[50%] h-[50%] rounded-full bg-purple-600/20 blur-[150px] animate-blob" style={{animationDelay: '2s'}}></div>
        <div className="absolute top-[40%] left-[60%] w-[30%] h-[30%] rounded-full bg-teal-600/10 blur-[100px] animate-blob" style={{animationDelay: '4s'}}></div>
      </div>
      
      <FloatingMath />

      {/* Навигация */}
      <nav className={`fixed top-0 w-full z-50 transition-all duration-500 ${scrolled ? 'bg-slate-950/50 backdrop-blur-xl border-b border-white/10 py-4 shadow-[0_4px_30px_rgba(0,0,0,0.3)]' : 'bg-transparent py-6'}`}>
        <div className="max-w-6xl mx-auto px-6 flex justify-between items-center">
          <div className="text-2xl font-bold tracking-tighter flex items-center gap-2 text-white drop-shadow-md group cursor-pointer">
            <svg className="w-6 h-6 text-blue-400 drop-shadow-[0_0_8px_rgba(96,165,250,0.8)] group-hover:rotate-90 transition-transform duration-500" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 2L2 22h20L12 2zm0 4.5l6.5 13h-13L12 6.5z"/>
            </svg>
            Pythagoras <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-teal-400">1.0</span>
          </div>
          <div className="hidden md:flex gap-8 text-sm font-medium text-slate-300">
            {['О проекте', 'Особенности', 'Архитектура', 'Начать'].map((item) => (
              <a key={item} href={`#${item === 'О проекте' ? 'about' : item === 'Особенности' ? 'features' : item === 'Архитектура' ? 'architecture' : 'get-started'}`} 
                 className="relative hover:text-white transition-colors group py-1">
                {item}
                <span className="absolute bottom-0 left-0 w-0 h-0.5 bg-gradient-to-r from-blue-400 to-purple-400 transition-all duration-300 group-hover:w-full"></span>
              </a>
            ))}
          </div>
        </div>
      </nav>

      {/* Hero Секция */}
      <section className="relative min-h-screen flex items-center pt-24 pb-16 z-10">
        <GridBackground />
        
        <div className="relative w-full max-w-6xl mx-auto px-6 grid lg:grid-cols-2 gap-12 items-center">
          <div className="text-left order-2 lg:order-1 z-10">
            <RevealOnScroll direction="left">
              <div className="inline-flex items-center gap-2 px-4 py-1.5 mb-6 rounded-full bg-white/5 backdrop-blur-md border border-white/10 text-sm font-medium text-blue-300 shadow-[0_4px_30px_rgba(0,0,0,0.1)] hover:bg-white/10 transition-colors">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-500"></span>
                </span>
                Образовательная LLM
              </div>
            </RevealOnScroll>
            
            <RevealOnScroll delay={100} direction="left">
              <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight mb-6 text-white leading-[1.1] drop-shadow-lg">
                Математика <br className="hidden lg:block" />
                встречает <br/>
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-purple-400 to-teal-400 drop-shadow-[0_0_15px_rgba(168,85,247,0.4)]">
                  Нейросети
                </span>
              </h1>
            </RevealOnScroll>
            
            <RevealOnScroll delay={200} direction="left">
              <p className="text-lg md:text-xl text-slate-300 mb-10 max-w-lg leading-relaxed drop-shadow-md">
                Минималистичная, но мощная языковая модель, созданная с нуля для решения базовых математических задач. Поймите каждый тензор и слой без скрытой магии.
              </p>
            </RevealOnScroll>
            
            <RevealOnScroll delay={300} direction="up">
              <div className="flex flex-col sm:flex-row gap-5">
                <a href="#get-started" className="relative group px-8 py-3.5 text-center rounded-xl bg-blue-600/80 backdrop-blur-md border border-blue-400/30 text-white font-semibold transition-all overflow-hidden shadow-[0_8px_32px_rgba(37,99,235,0.4)] hover:shadow-[0_8px_32px_rgba(37,99,235,0.6)] hover:-translate-y-1">
                  <span className="relative z-10">Начать работу</span>
                  <div className="absolute inset-0 h-full w-full scale-0 rounded-xl transition-all duration-300 ease-out group-hover:scale-100 group-hover:bg-blue-500/50"></div>
                </a>
                <a href="https://github.com/zzzigrok/Pythagoras-1.0" target="_blank" rel="noreferrer" className="px-8 py-3.5 text-center rounded-xl bg-white/5 hover:bg-white/10 backdrop-blur-md border border-white/10 text-white font-medium transition-all flex items-center justify-center gap-2 shadow-[0_8px_32px_rgba(0,0,0,0.2)] hover:border-white/20 hover:-translate-y-1">
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
                  GitHub
                </a>
              </div>
            </RevealOnScroll>
          </div>
          
          <div className="order-1 lg:order-2 flex justify-center items-center relative z-10">
             <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-72 h-72 bg-blue-500/20 rounded-full blur-[80px] pointer-events-none animate-blob"></div>
             <RevealOnScroll delay={400} direction="right">
                <AnimatedTriangle />
             </RevealOnScroll>
          </div>
        </div>
      </section>

      {/* О проекте */}
      <div id="about" className="relative z-10">
        <Section title="Философия Проекта">
          <RevealOnScroll delay={100}>
            <div className="grid lg:grid-cols-2 gap-12 items-center bg-white/5 backdrop-blur-xl border border-white/10 rounded-3xl p-8 md:p-12 shadow-[0_8px_32px_rgba(0,0,0,0.2)] hover:border-white/20 transition-colors duration-500">
              <div className="space-y-6">
                <p className="text-lg text-slate-300 leading-relaxed">
                  Pythagoras 1.0 создан не для того, чтобы конкурировать с гигантскими моделями от OpenAI или Google. Его цель — <span className="text-white font-medium bg-blue-500/20 border border-blue-500/30 px-2 py-1 rounded-md backdrop-blur-sm shadow-[0_0_10px_rgba(59,130,246,0.2)]">образовательная прозрачность</span>.
                </p>
                <p className="text-lg text-slate-300 leading-relaxed">
                  Мы написали LLM полностью с нуля на PyTorch, убрав все лишние абстракции. Вы можете проследить путь тензора от входного токена до финального предсказания.
                </p>
                <ul className="space-y-4 text-slate-300 mt-8">
                  {[
                    { color: 'blue', text: 'Никаких скрытых библиотек вроде HuggingFace' },
                    { color: 'purple', text: 'Фокус на архитектуре Transformer' },
                    { color: 'teal', text: 'Обучена на простых математических операциях' }
                  ].map((item, i) => (
                    <RevealOnScroll key={i} delay={300 + (i * 150)} direction="left">
                      <li className="flex items-center gap-4 bg-white/5 p-4 rounded-2xl border border-white/5 backdrop-blur-sm hover:bg-white/10 transition-colors">
                        <span className={`w-3 h-3 rounded-full bg-${item.color}-400 shadow-[0_0_12px_rgba(var(--tw-colors-${item.color}-400),0.8)] block animate-pulse`}></span>
                        <span className="font-medium text-slate-200">{item.text}</span>
                      </li>
                    </RevealOnScroll>
                  ))}
                </ul>
              </div>
              
              {/* Анимированная иллюстрация обучения (Glassmorphism + Пути) */}
              <div className="relative h-96 bg-slate-900/40 backdrop-blur-md rounded-2xl border border-white/10 overflow-hidden flex items-center justify-center p-6 shadow-[inset_0_0_30px_rgba(255,255,255,0.02)]">
                <div className="absolute inset-0 bg-gradient-to-br from-blue-500/10 to-purple-500/10 mix-blend-overlay"></div>
                
                <svg className="w-full h-full drop-shadow-xl" viewBox="0 0 400 200">
                  <defs>
                    <path id="path1" d="M90 50 Q 105 100 120 100" />
                    <path id="path2" d="M160 100 L 180 100" />
                    <path id="path3" d="M220 100 L 240 100" />
                    <path id="path4" d="M280 100 Q 295 100 310 150" />
                    
                    <filter id="neonGlow" x="-20%" y="-20%" width="140%" height="140%">
                      <feGaussianBlur stdDeviation="3" result="blur" />
                      <feComposite in="SourceGraphic" in2="blur" operator="over" />
                    </filter>
                  </defs>

                  {/* Входные данные */}
                  <rect x="20" y="35" width="70" height="30" rx="8" fill="rgba(59,130,246,0.1)" stroke="rgba(59,130,246,0.4)" strokeWidth="1.5" />
                  <text x="55" y="54" fill="#60a5fa" fontSize="12" textAnchor="middle" className="font-mono font-bold">2 + 2 =</text>
                  
                  {/* Слои сети (стеклянные блоки) с пульсацией обводки */}
                  <rect x="120" y="80" width="40" height="40" rx="10" fill="rgba(59,130,246,0.15)" stroke="rgba(59,130,246,0.6)" strokeWidth="2" filter="url(#neonGlow)">
                      <animate attributeName="stroke-opacity" values="0.4;1;0.4" dur="2s" repeatCount="indefinite"/>
                  </rect>
                  <rect x="180" y="80" width="40" height="40" rx="10" fill="rgba(168,85,247,0.15)" stroke="rgba(168,85,247,0.6)" strokeWidth="2" filter="url(#neonGlow)">
                      <animate attributeName="stroke-opacity" values="0.4;1;0.4" dur="2s" repeatCount="indefinite" begin="0.5s"/>
                  </rect>
                  <rect x="240" y="80" width="40" height="40" rx="10" fill="rgba(45,212,191,0.15)" stroke="rgba(45,212,191,0.6)" strokeWidth="2" filter="url(#neonGlow)">
                      <animate attributeName="stroke-opacity" values="0.4;1;0.4" dur="2s" repeatCount="indefinite" begin="1s"/>
                  </rect>
                  
                  {/* Базовые линии соединений с бегущей линией (dasharray) */}
                  <use href="#path1" fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="3" strokeLinecap="round" />
                  <use href="#path1" fill="none" stroke="rgba(96,165,250,0.5)" strokeWidth="1.5" strokeLinecap="round" strokeDasharray="5, 10">
                     <animate attributeName="stroke-dashoffset" from="15" to="0" dur="1s" repeatCount="indefinite" />
                  </use>

                  <use href="#path2" fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="3" strokeLinecap="round" />
                  <use href="#path2" fill="none" stroke="rgba(192,132,252,0.5)" strokeWidth="1.5" strokeLinecap="round" strokeDasharray="5, 10">
                     <animate attributeName="stroke-dashoffset" from="15" to="0" dur="1s" repeatCount="indefinite" />
                  </use>

                  <use href="#path3" fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="3" strokeLinecap="round" />
                  <use href="#path3" fill="none" stroke="rgba(45,212,191,0.5)" strokeWidth="1.5" strokeLinecap="round" strokeDasharray="5, 10">
                     <animate attributeName="stroke-dashoffset" from="15" to="0" dur="1s" repeatCount="indefinite" />
                  </use>

                  <use href="#path4" fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="3" strokeLinecap="round" />
                   <use href="#path4" fill="none" stroke="rgba(52,211,153,0.5)" strokeWidth="1.5" strokeLinecap="round" strokeDasharray="5, 10">
                     <animate attributeName="stroke-dashoffset" from="15" to="0" dur="1s" repeatCount="indefinite" />
                  </use>

                  {/* Бегущие пакеты данных (точки) */}
                  <circle r="3" fill="#ffffff" filter="url(#neonGlow)">
                    <animateMotion dur="2s" repeatCount="indefinite" keyPoints="0;1" keyTimes="0;1" calcMode="linear">
                      <mpath href="#path1" />
                    </animateMotion>
                  </circle>
                  <circle r="3" fill="#ffffff" filter="url(#neonGlow)">
                    <animateMotion dur="2s" repeatCount="indefinite" begin="0.5s" keyPoints="0;1" keyTimes="0;1" calcMode="linear">
                      <mpath href="#path2" />
                    </animateMotion>
                  </circle>
                  <circle r="3" fill="#ffffff" filter="url(#neonGlow)">
                    <animateMotion dur="2s" repeatCount="indefinite" begin="1s" keyPoints="0;1" keyTimes="0;1" calcMode="linear">
                      <mpath href="#path3" />
                    </animateMotion>
                  </circle>
                  <circle r="3" fill="#ffffff" filter="url(#neonGlow)">
                    <animateMotion dur="2s" repeatCount="indefinite" begin="1.5s" keyPoints="0;1" keyTimes="0;1" calcMode="linear">
                      <mpath href="#path4" />
                    </animateMotion>
                  </circle>

                  {/* Выход */}
                  <rect x="290" y="135" width="40" height="30" rx="8" fill="rgba(16,185,129,0.1)" stroke="rgba(16,185,129,0.5)" strokeWidth="1.5" filter="url(#neonGlow)"/>
                  <text x="310" y="155" fill="#34d399" fontSize="16" textAnchor="middle" className="font-mono font-bold animate-pulse">4</text>

                  {/* Подписи слоев */}
                  <text x="140" y="140" fill="#94a3b8" fontSize="11" textAnchor="middle" className="font-medium tracking-widest">EMB</text>
                  <text x="200" y="140" fill="#94a3b8" fontSize="11" textAnchor="middle" className="font-medium tracking-widest">ATTN</text>
                  <text x="260" y="140" fill="#94a3b8" fontSize="11" textAnchor="middle" className="font-medium tracking-widest">FFN</text>
                </svg>
              </div>
            </div>
          </RevealOnScroll>
        </Section>
      </div>

      {/* Особенности */}
      <div id="features" className="relative z-10">
        <Section title="Ключевые Особенности">
          <div className="grid md:grid-cols-3 gap-8">
            {[
              { id: 'brain', title: 'Чистый PyTorch', color: 'blue', desc: 'Модель реализована с использованием базовых тензорных операций. Идеально для изучения механики Self-Attention и Transformer блоков.' },
              { id: 'code', title: 'Легковесность', color: 'purple', desc: 'Архитектура спроектирована так, чтобы обучение и инференс можно было запускать даже на обычном CPU без мощных графических ускорителей.' },
              { id: 'lightning', title: 'Обучающий набор', color: 'teal', desc: 'Включает генератор кастомных датасетов для математических задач (сложение, вычитание, умножение), позволяя экспериментировать с данными.' }
            ].map((feature, idx) => (
              <RevealOnScroll key={feature.id} delay={idx * 150} direction="up">
                <div className="bg-white/5 backdrop-blur-lg p-8 rounded-3xl border border-white/10 hover:bg-white/10 transition-all duration-500 group shadow-[0_8px_32px_rgba(0,0,0,0.1)] hover:shadow-[0_15px_35px_rgba(0,0,0,0.3)] hover:-translate-y-3 h-full">
                  <div className={`w-16 h-16 bg-${feature.color}-500/20 backdrop-blur-md rounded-2xl flex items-center justify-center text-${feature.color}-400 mb-8 group-hover:scale-110 group-hover:rotate-6 transition-all duration-300 border border-${feature.color}-500/30 shadow-[0_0_20px_rgba(var(--tw-colors-${feature.color}-500),0.3)]`}>
                    <FeatureIcon type={feature.id} />
                  </div>
                  <h3 className="text-2xl font-bold mb-4 text-white drop-shadow-sm">{feature.title}</h3>
                  <p className="text-slate-300 leading-relaxed text-base opacity-80 group-hover:opacity-100 transition-opacity">
                    {feature.desc}
                  </p>
                </div>
              </RevealOnScroll>
            ))}
          </div>
        </Section>
      </div>

      {/* Архитектура */}
      <div id="architecture" className="relative z-10">
        <Section title="Структура Проекта">
          <RevealOnScroll delay={100}>
            <div className="max-w-4xl mx-auto bg-[#0f172a]/80 backdrop-blur-2xl rounded-3xl border border-white/10 p-8 md:p-10 font-mono text-sm shadow-[0_20px_50px_rgba(0,0,0,0.4)] overflow-hidden relative">
              
              {/* Свечение терминала */}
              <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full h-1 bg-gradient-to-r from-transparent via-blue-500/50 to-transparent"></div>
              
              <div className="flex items-center gap-2 mb-8 border-b border-white/10 pb-6">
                <div className="w-3.5 h-3.5 rounded-full bg-red-500/90 shadow-[0_0_10px_rgba(239,68,68,0.6)]"></div>
                <div className="w-3.5 h-3.5 rounded-full bg-yellow-500/90 shadow-[0_0_10px_rgba(234,179,8,0.6)]"></div>
                <div className="w-3.5 h-3.5 rounded-full bg-green-500/90 shadow-[0_0_10px_rgba(34,197,94,0.6)]"></div>
                <span className="ml-4 text-slate-400 font-medium select-none bg-white/5 px-3 py-1.5 rounded-lg backdrop-blur-sm flex items-center gap-2">
                  <svg className="w-4 h-4 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"/></svg>
                  pythagoras-1.0/
                </span>
              </div>
              
              <div className="space-y-4 text-slate-300 ml-2 md:text-base">
                {[
                  { icon: '📁', color: 'blue', name: 'data/', desc: '# Датасеты (input_math.txt)' },
                  { icon: '📁', color: 'blue', name: 'docs/', desc: '# Документация и туториалы' },
                  { indent: true, icon: '📁', color: 'slate', name: 'subdocs/', desc: '# Описание компонентов' },
                  { indent: true, icon: '📁', color: 'slate', name: 'tutorials/', desc: '# Пошаговые руководства', isLastChild: true },
                  { icon: '📁', color: 'blue', name: 'weights/', desc: '# Сохраненные веса (.pth)' },
                  { icon: '🐍', color: 'green', name: 'pythagoras_hub.py', desc: '# Главный скрипт', glow: true },
                  { icon: '📄', color: 'green', name: 'requirements.txt', desc: '# Зависимости (torch)', isLast: true }
                ].map((item, idx) => (
                  <RevealOnScroll key={idx} delay={200 + (idx * 100)} direction="left">
                    <div className={`flex group items-center hover:bg-white/5 p-2 rounded-xl transition-all duration-300 cursor-default ${item.indent ? 'pl-10' : ''}`}>
                      <span className={`text-${item.color}-${item.indent ? '500' : '400'} mr-4 font-bold`}>{item.isLast ? '└──' : item.isLastChild ? '└──' : '├──'}</span> 
                      <span className="mr-2 opacity-80">{item.icon}</span>
                      <span className={`text-${item.color}-300 font-medium tracking-wide ${item.glow ? 'drop-shadow-[0_0_8px_rgba(74,222,128,0.6)] text-green-300' : ''}`}>{item.name}</span>
                      <span className="ml-auto md:ml-8 text-slate-500 text-xs md:text-sm italic opacity-0 group-hover:opacity-100 transition-opacity duration-300 transform translate-x-4 group-hover:translate-x-0">{item.desc}</span>
                    </div>
                  </RevealOnScroll>
                ))}
              </div>
            </div>
          </RevealOnScroll>
        </Section>
      </div>

      {/* Как начать */}
      <div id="get-started" className="relative z-10 pb-20">
        <Section title="Быстрый Старт">
          <div className="max-w-3xl mx-auto space-y-10">
            
            <RevealOnScroll delay={100} direction="up">
              <div className="bg-white/5 backdrop-blur-xl rounded-3xl border border-white/10 overflow-hidden shadow-[0_15px_35px_rgba(0,0,0,0.3)] hover:border-white/20 transition-colors duration-500">
                <div className="bg-white/5 px-6 py-4 border-b border-white/10 flex items-center justify-between">
                  <span className="text-sm text-slate-200 font-bold flex items-center gap-3">
                    <span className="flex items-center justify-center w-8 h-8 rounded-full bg-blue-500/20 text-blue-400 text-sm border border-blue-500/30 shadow-[0_0_10px_rgba(59,130,246,0.3)]">1</span>
                    Установка зависимостей
                  </span>
                  <div className="flex gap-1.5">
                    <div className="w-2.5 h-2.5 rounded-full bg-slate-600"></div>
                    <div className="w-2.5 h-2.5 rounded-full bg-slate-600"></div>
                  </div>
                </div>
                <div className="p-6 bg-[#0B1120]/80 font-mono text-sm md:text-base text-blue-300">
                  <code className="flex items-center gap-3">
                    <span className="text-slate-500 select-none">$</span>
                    pip install -r requirements.txt
                  </code>
                </div>
              </div>
            </RevealOnScroll>

            <RevealOnScroll delay={200} direction="up">
              <div className="bg-white/5 backdrop-blur-xl rounded-3xl border border-white/10 overflow-hidden shadow-[0_15px_35px_rgba(0,0,0,0.3)] hover:border-white/20 transition-colors duration-500">
                <div className="bg-white/5 px-6 py-4 border-b border-white/10 flex items-center justify-between">
                  <span className="text-sm text-slate-200 font-bold flex items-center gap-3">
                    <span className="flex items-center justify-center w-8 h-8 rounded-full bg-purple-500/20 text-purple-400 text-sm border border-purple-500/30 shadow-[0_0_10px_rgba(168,85,247,0.3)]">2</span>
                    Использование готовой модели
                  </span>
                  <div className="text-xs text-slate-500 font-mono">python</div>
                </div>
                <div className="p-6 bg-[#0B1120]/80 font-mono text-sm md:text-base text-slate-300 overflow-x-auto whitespace-pre leading-loose">
<span className="text-purple-400">from</span> pythagoras_hub <span className="text-purple-400">import</span> SimpleLLM, generate_text
<span className="text-purple-400">import</span> torch
<span className="text-purple-400">import</span> pickle

<span className="text-slate-500 italic"># Загрузка словаря</span>
<span className="text-purple-400">with</span> <span className="text-blue-300">open</span>(<span className="text-teal-300">'weights/math_vocab.pkl'</span>, <span className="text-teal-300">'rb'</span>) <span className="text-purple-400">as</span> f:
    vocab = pickle.load(f)
    
<span className="text-slate-500 italic"># Инициализация модели</span>
model = SimpleLLM(vocab_size=<span className="text-blue-300">len</span>(vocab))
model.load_state_dict(torch.load(<span className="text-teal-300">'weights/math_model_weights.pth'</span>))

<span className="text-slate-500 italic"># Генерация решения</span>
<span className="text-blue-300">print</span>(generate_text(model, <span className="text-teal-300">"12 + 5 = "</span>, vocab))
                </div>
              </div>
            </RevealOnScroll>
            
            <RevealOnScroll delay={300}>
              <div className="text-center pt-8">
                <a href="https://github.com/zzzigrok/Pythagoras-1.0/tree/main/docs" target="_blank" rel="noreferrer" className="group inline-flex items-center gap-3 px-8 py-4 rounded-full bg-white/5 hover:bg-blue-600/20 backdrop-blur-md border border-white/10 hover:border-blue-400/50 text-blue-300 hover:text-white transition-all duration-300 shadow-lg hover:shadow-[0_0_30px_rgba(59,130,246,0.3)]">
                  <span className="font-semibold">Читать полную документацию</span>
                  <svg className="w-5 h-5 transform group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3"></path></svg>
                </a>
              </div>
            </RevealOnScroll>

          </div>
        </Section>
      </div>

      {/* Footer */}
      <footer className="relative z-10 bg-[#020617]/80 backdrop-blur-2xl border-t border-white/10 py-16 text-center text-slate-400 overflow-hidden">
        <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-full h-[1px] bg-gradient-to-r from-transparent via-blue-500/50 to-transparent"></div>
        <RevealOnScroll>
          <div className="max-w-6xl mx-auto px-6">
            <div className="flex items-center justify-center gap-3 mb-8 opacity-80 hover:opacity-100 transition-opacity">
               <svg className="w-8 h-8 text-blue-500 drop-shadow-md" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 2L2 22h20L12 2zm0 4.5l6.5 13h-13L12 6.5z"/>
              </svg>
              <span className="font-bold text-white tracking-widest text-xl drop-shadow-sm">PYTHAGORAS <span className="text-blue-500">1.0</span></span>
            </div>
            <p className="mb-3 font-medium text-slate-300 text-lg">Обучающая языковая модель на PyTorch</p>
            <p className="text-sm opacity-50 tracking-wide uppercase">Распространяется по лицензии MIT</p>
          </div>
        </RevealOnScroll>
      </footer>
    </div>
  );
}
