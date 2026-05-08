/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef, useMemo } from 'react';
import { 
  Activity, 
  Cpu, 
  Database, 
  Zap, 
  Layers, 
  HardDrive, 
  Terminal,
  TrendingUp,
  BarChart3,
  Clock
} from 'lucide-react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  AreaChart,
  Area
} from 'recharts';
import { motion, AnimatePresence } from 'motion/react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// --- Types & Constants ---
interface Metric {
  time: number;
  latency: number;
  throughput: number;
  cacheHitRate: number;
}

interface LogEntry {
  id: string;
  timestamp: string;
  message: string;
  type: 'info' | 'success' | 'warning' | 'error';
}

const MAX_EXPERTS = 64;
const CACHE_SIZE = 16;
const NVME_BASE_LATENCY_MS = 2.5; // High-end NVMe PCIe Gen 4/5
const INTERCONNECT_LATENCY_MS = 0.5;
const RAM_LATENCY_MS = 0.05;

export default function App() {
  const [metrics, setMetrics] = useState<Metric[]>([]);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [activeExperts, setActiveExperts] = useState<number[]>([]);
  const [cache, setCache] = useState<number[]>([]);
  const [isSimulating, setIsSimulating] = useState(false);
  const [tokenCount, setTokenCount] = useState(0);
  
  const simulationRef = useRef<NodeJS.Timeout | null>(null);

  const addLog = (message: string, type: LogEntry['type'] = 'info') => {
    setLogs(prev => [
      {
        id: Math.random().toString(36).substr(2, 9),
        timestamp: new Date().toLocaleTimeString(),
        message,
        type
      },
      ...prev.slice(0, 49)
    ]);
  };

  const simulateStep = () => {
    // 1. Router selects 2 experts (Top-K)
    const target1 = Math.floor(Math.random() * MAX_EXPERTS);
    const target2 = (target1 + Math.floor(Math.random() * 5) + 1) % MAX_EXPERTS;
    const selected = [target1, target2];
    setActiveExperts(selected);

    let totalLatency = INTERCONNECT_LATENCY_MS;
    let hits = 0;

    selected.forEach(id => {
      setCache(prev => {
        if (prev.includes(id)) {
          hits++;
          totalLatency += RAM_LATENCY_MS;
          addLog(`Expert ${id}: Cache Hit (RAM)`, 'success');
          // Move to front (LRU)
          return [id, ...prev.filter(x => x !== id)];
        } else {
          totalLatency += NVME_BASE_LATENCY_MS + (Math.random() * 1.5);
          addLog(`Expert ${id}: Cache Miss -> Fetching via io_uring (NVMe)`, 'warning');
          const newCache = [id, ...prev.filter(x => x !== id)];
          return newCache.slice(0, CACHE_SIZE);
        }
      });
    });

    setTokenCount(prev => prev + 1);
    
    // Performance stats
    const currentMetric: Metric = {
      time: Date.now(),
      latency: totalLatency,
      throughput: 1000 / totalLatency,
      cacheHitRate: (hits / 2) * 100
    };

    setMetrics(prev => [...prev.slice(-29), currentMetric]);
  };

  useEffect(() => {
    if (isSimulating) {
      simulationRef.current = setInterval(simulateStep, 800);
    } else {
      if (simulationRef.current) clearInterval(simulationRef.current);
    }
    return () => {
      if (simulationRef.current) clearInterval(simulationRef.current);
    };
  }, [isSimulating]);

  const avgLatency = useMemo(() => {
    if (metrics.length === 0) return 0;
    return metrics.reduce((acc, m) => acc + m.latency, 0) / metrics.length;
  }, [metrics]);

  const avgThroughput = useMemo(() => {
    if (metrics.length === 0) return 0;
    return metrics.reduce((acc, m) => acc + m.throughput, 0) / metrics.length;
  }, [metrics]);

  return (
    <div id="system-dashboard" className="min-h-screen bg-[#0a0a0c] text-[#e1e1e6] font-sans selection:bg-blue-500/30">
      {/* Sidebar / Navigation */}
      <nav className="fixed left-0 top-0 bottom-0 w-20 border-r border-white/5 bg-black/40 backdrop-blur-xl flex flex-col items-center py-8 gap-10 z-50">
        <div className="bg-blue-600 p-3 rounded-xl shadow-[0_0_20px_rgba(37,99,235,0.4)]">
          <Layers className="w-6 h-6 text-white" />
        </div>
        <div className="flex flex-col gap-8 opacity-40">
          <Activity className="w-6 h-6 hover:text-blue-400 cursor-pointer transition-colors" />
          <Database className="w-6 h-6 hover:text-blue-400 cursor-pointer transition-colors" />
          <Cpu className="w-6 h-6 hover:text-blue-400 cursor-pointer transition-colors" />
          <SettingsIcon className="w-6 h-6 hover:text-blue-400 cursor-pointer transition-colors" />
        </div>
      </nav>

      <main className="pl-20 p-8 max-w-[1600px] mx-auto">
        {/* Header */}
        <header className="mb-12 flex justify-between items-end">
          <div>
            <div className="flex items-center gap-2 text-blue-400 font-mono text-sm mb-2">
              <Zap className="w-4 h-4 fill-current" />
              <span>SYSTEM LIVE: PCI-E 5.0 NVME INTERFACE</span>
            </div>
            <h1 className="text-4xl font-bold tracking-tight bg-gradient-to-r from-white to-white/40 bg-clip-text text-transparent">
              Micro-Expert Router
            </h1>
          </div>
          <div className="flex gap-4">
            <button 
              onClick={() => setIsSimulating(!isSimulating)}
              className={cn(
                "px-6 py-3 rounded-lg font-medium transition-all duration-300 flex items-center gap-2 border",
                isSimulating 
                  ? "bg-red-500/10 border-red-500/50 text-red-400 hover:bg-red-500/20" 
                  : "bg-blue-600 border-blue-400 text-white hover:bg-blue-700 shadow-lg shadow-blue-500/20"
              )}
            >
              {isSimulating ? (
                <><Square className="w-4 h-4 fill-current" /> Terminate Stream</>
              ) : (
                <><Play className="w-4 h-4 fill-current" /> Start Inference Stream</>
              )}
            </button>
          </div>
        </header>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <StatCard 
            icon={<Clock className="w-5 h-5 text-blue-400" />}
            label="Avg E2E Latency"
            value={`${avgLatency.toFixed(2)}ms`}
            subValue="Target: <5.00ms"
            trend={-12}
          />
          <StatCard 
            icon={<TrendingUp className="w-5 h-5 text-emerald-400" />}
            label="Throughput"
            value={`${avgThroughput.toFixed(1)} T/s`}
            subValue="P99 Optimized"
            trend={8.4}
          />
          <StatCard 
            icon={<Database className="w-5 h-5 text-purple-400" />}
            label="Cache Hit Rate"
            value={`${metrics.length > 0 ? metrics[metrics.length-1].cacheHitRate.toFixed(0) : 0}%`}
            subValue={`LRU: ${CACHE_SIZE}/${MAX_EXPERTS} SLOTS`}
            trend={0}
          />
          <StatCard 
            icon={<HardDrive className="w-5 h-5 text-orange-400" />}
            label="NVMe I/O Intensity"
            value="High"
            subValue="Direct I/O Enabled"
            trend={100}
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Chart */}
          <div className="lg:col-span-2 space-y-8">
            <section className="bg-white/[0.03] border border-white/5 rounded-2xl p-6 backdrop-blur-3xl">
              <div className="flex items-center justify-between mb-8">
                <div className="flex items-center gap-3">
                  <BarChart3 className="w-5 h-5 text-blue-400" />
                  <h2 className="font-semibold text-lg">Inference Performance Matrix</h2>
                </div>
                <div className="flex items-center gap-4 text-xs font-mono opacity-50">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-blue-500" /> Latency (ms)
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-emerald-500" /> Throughput (T/s)
                  </div>
                </div>
              </div>
              <div className="h-[350px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={metrics}>
                    <defs>
                      <linearGradient id="colorLatency" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#ffffff05" />
                    <XAxis 
                      hide
                      dataKey="time" 
                    />
                    <YAxis 
                      stroke="#ffffff20" 
                      fontSize={12} 
                      tickFormatter={(v) => `${v}`}
                    />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1a1a1f', border: '1px solid #ffffff10', borderRadius: '8px' }}
                      itemStyle={{ fontSize: '12px' }}
                    />
                    <Area 
                      type="monotone" 
                      dataKey="latency" 
                      stroke="#3b82f6" 
                      strokeWidth={2}
                      fillOpacity={1} 
                      fill="url(#colorLatency)" 
                      animationDuration={400}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="throughput" 
                      stroke="#10b981" 
                      strokeWidth={2} 
                      dot={false}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </section>

            {/* Expert Map */}
            <section className="bg-white/[0.03] border border-white/5 rounded-2xl p-6">
              <div className="flex items-center gap-3 mb-6">
                <Layers className="w-5 h-5 text-blue-400" />
                <h2 className="font-semibold text-lg">Expert Shard Visualization (Active: {activeExperts.join(', ')})</h2>
              </div>
              <div className="grid grid-cols-16 gap-2">
                {Array.from({ length: MAX_EXPERTS }).map((_, i) => (
                  <ExpertNode 
                    key={i} 
                    id={i} 
                    isActive={activeExperts.includes(i)} 
                    isCached={cache.includes(i)}
                  />
                ))}
              </div>
            </section>
          </div>

          {/* Console / Logs */}
          <div className="space-y-8">
            <section className="bg-black border border-white/5 rounded-2xl flex flex-col h-full min-h-[600px]">
              <div className="p-4 border-b border-white/5 flex items-center justify-between bg-white/[0.02]">
                <div className="flex items-center gap-2">
                  <Terminal className="w-4 h-4 text-emerald-400" />
                  <span className="text-xs font-mono font-medium tracking-tight">IO_URING KERNEL LOGS</span>
                </div>
                <div className="flex gap-1">
                  <div className="w-2 h-2 rounded-full bg-red-500/20" />
                  <div className="w-2 h-2 rounded-full bg-yellow-500/20" />
                  <div className="w-2 h-2 rounded-full bg-green-500/20" />
                </div>
              </div>
              <div className="flex-1 p-4 font-mono text-[11px] overflow-y-auto space-y-2 scrollbar-thin scrollbar-thumb-white/10">
                <AnimatePresence initial={false}>
                  {logs.map((log) => (
                    <motion.div
                      key={log.id}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      className={cn(
                        "flex gap-3 leading-relaxed",
                        log.type === 'success' && "text-emerald-400/80",
                        log.type === 'warning' && "text-amber-400/80",
                        log.type === 'error' && "text-red-400/80",
                        log.type === 'info' && "text-white/40"
                      )}
                    >
                      <span className="opacity-30 shrink-0">[{log.timestamp}]</span>
                      <span>{log.message}</span>
                    </motion.div>
                  ))}
                  {logs.length === 0 && (
                    <div className="text-white/20 italic">Waiting for connection to NVMe controller...</div>
                  )}
                </AnimatePresence>
              </div>
            </section>
          </div>
        </div>
      </main>
    </div>
  );
}

// --- Subcomponents ---

function StatCard({ icon, label, value, subValue, trend }: { 
  icon: React.ReactNode, 
  label: string, 
  value: string, 
  subValue: string,
  trend: number
}) {
  return (
    <div className="bg-white/[0.03] border border-white/5 rounded-xl p-5 backdrop-blur-sm relative overflow-hidden group">
      <div className="absolute top-0 right-0 w-24 h-24 bg-blue-500/5 blur-3xl rounded-full -mr-12 -mt-12 group-hover:bg-blue-500/10 transition-colors" />
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 rounded-lg bg-white/5 border border-white/10">
          {icon}
        </div>
        <span className="text-xs font-medium text-white/50 uppercase tracking-widest">{label}</span>
      </div>
      <div className="flex items-baseline gap-2 mb-1">
        <span className="text-2xl font-bold font-mono tracking-tight">{value}</span>
        {trend !== 0 && (
          <span className={cn("text-[10px] font-bold px-1.5 py-0.5 rounded", 
            trend > 0 ? "bg-emerald-500/10 text-emerald-400" : "bg-red-500/10 text-red-400"
          )}>
            {trend > 0 ? '+' : ''}{trend}%
          </span>
        )}
      </div>
      <span className="text-[10px] font-mono text-white/30">{subValue}</span>
    </div>
  );
}

function ExpertNode({ id, isActive, isCached }: { id: number; isActive: boolean; isCached: boolean }) {
  return (
    <div 
      className={cn(
        "aspect-square rounded-sm transition-all duration-300 relative",
        isActive 
          ? "bg-blue-500 shadow-[0_0_15px_#3b82f6] border-blue-400 z-10 scale-110" 
          : isCached 
            ? "bg-emerald-500/40 border border-emerald-500/30" 
            : "bg-white/5 border border-white/5 hover:border-white/20"
      )}
      title={`Expert ID: ${id} | ${isCached ? 'Cached' : 'On Disk'}`}
    >
      {isActive && (
        <span className="absolute -top-6 left-1/2 -translate-x-1/2 text-[10px] font-bold text-blue-400 whitespace-nowrap animate-pulse">
          IO_REQ
        </span>
      )}
    </div>
  );
}

function SettingsIcon(props: any) {
  return (
    <svg 
      {...props} 
      xmlns="http://www.w3.org/2000/svg" 
      width="24" 
      height="24" 
      viewBox="0 0 24 24" 
      fill="none" 
      stroke="currentColor" 
      strokeWidth="2" 
      strokeLinecap="round" 
      strokeLinejoin="round"
    >
      <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  );
}

function Play(props: any) {
  return (
    <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>
  );
}

function Square(props: any) {
  return (
    <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect width="18" height="18" x="3" y="3" rx="2"/></svg>
  );
}
