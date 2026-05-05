import { useState, useEffect } from 'react';
import { Play, UserPlus, ShieldAlert, ShieldCheck, Zap, Activity } from 'lucide-react';
import './Sandbox.css';

const API = 'http://localhost:8000/api';

const DEFAULT_TARGET = {
  sent: 1000,
  received: 5000,
  trades_out: 2,
  trades_in: 5
};

const HACKER_TEMPLATE = {
  sent: 5000000,
  received: 0,
  trades_out: 50,
  trades_in: 0
};

const SAFE_TEMPLATE = {
  sent: 200,
  received: 300,
  trades_out: 1,
  trades_in: 1
};

export default function Sandbox() {
  const [target, setTarget] = useState(DEFAULT_TARGET);
  const [neighbors, setNeighbors] = useState([]);
  const [riskScore, setRiskScore] = useState(0);
  const [loading, setLoading] = useState(false);

  const simulate = async (currentTarget, currentNeighbors) => {
    setLoading(true);
    try {
      const res = await fetch(`${API}/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          target: currentTarget,
          neighbors: currentNeighbors
        })
      });
      const data = await res.json();
      if (data.risk_score !== undefined) {
        setRiskScore(data.risk_score);
      }
    } catch (err) {
      console.error(err);
    }
    setLoading(false);
  };

  // Run simulation on initial load and whenever network changes
  useEffect(() => {
    simulate(target, neighbors);
  }, [target, neighbors]);

  const updateTarget = (field, value) => {
    setTarget(prev => ({ ...prev, [field]: Number(value) }));
  };

  const addNeighbor = (type) => {
    const newNeighbor = type === 'hacker' ? { ...HACKER_TEMPLATE } : { ...SAFE_TEMPLATE };
    // Add small random noise so they aren't identical
    newNeighbor.sent += Math.floor(Math.random() * 100);
    setNeighbors(prev => [...prev, newNeighbor]);
  };

  const clearNeighbors = () => setNeighbors([]);

  const isHighRisk = riskScore > 0.8;
  const isMediumRisk = riskScore > 0.4;
  
  const scoreColor = isHighRisk ? '#f87171' : isMediumRisk ? '#fbbf24' : '#34d399';

  return (
    <div className="flex flex-col h-full overflow-hidden bg-surface-dim">
      <div className="flex flex-1 overflow-hidden">
        {/* Left Panel: Controls */}
        <section className="w-80 lg:w-96 border-r border-outline-variant/30 flex flex-col h-full glass-panel z-10 shrink-0">
          <div className="p-4 border-b border-outline-variant/20 flex items-center gap-2">
            <Zap className="text-primary" size={20} />
            <h2 className="font-h2 text-xl text-on-surface">Model Parameters</h2>
          </div>
          
          <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-6">
            <div className="flex flex-col gap-4">
              <h3 className="font-label-caps text-xs text-on-surface-variant uppercase tracking-widest">Input Features</h3>
              
              {/* Sent Slider */}
              <div className="flex flex-col gap-2">
                <div className="flex justify-between items-center">
                  <label className="font-data-mono text-sm text-on-surface">Total Sent</label>
                  <span className="font-data-mono text-primary">${target.sent.toLocaleString()}</span>
                </div>
                <input 
                  type="range" min="0" max="10000000" step="1000"
                  value={target.sent} onChange={e => updateTarget('sent', e.target.value)}
                  className="w-full h-1 bg-surface-container-high rounded-full appearance-none outline-none accent-primary"
                />
              </div>

              {/* Received Slider */}
              <div className="flex flex-col gap-2">
                <div className="flex justify-between items-center">
                  <label className="font-data-mono text-sm text-on-surface">Total Received</label>
                  <span className="font-data-mono text-primary">${target.received.toLocaleString()}</span>
                </div>
                <input 
                  type="range" min="0" max="10000000" step="1000"
                  value={target.received} onChange={e => updateTarget('received', e.target.value)}
                  className="w-full h-1 bg-surface-container-high rounded-full appearance-none outline-none accent-primary"
                />
              </div>

              {/* Trades Out Slider */}
              <div className="flex flex-col gap-2">
                <div className="flex justify-between items-center">
                  <label className="font-data-mono text-sm text-on-surface">Trades Out</label>
                  <span className="font-data-mono text-primary">{target.trades_out}</span>
                </div>
                <input 
                  type="range" min="0" max="100" step="1"
                  value={target.trades_out} onChange={e => updateTarget('trades_out', e.target.value)}
                  className="w-full h-1 bg-surface-container-high rounded-full appearance-none outline-none accent-primary"
                />
              </div>

              {/* Trades In Slider */}
              <div className="flex flex-col gap-2">
                <div className="flex justify-between items-center">
                  <label className="font-data-mono text-sm text-on-surface">Trades In</label>
                  <span className="font-data-mono text-primary">{target.trades_in}</span>
                </div>
                <input 
                  type="range" min="0" max="100" step="1"
                  value={target.trades_in} onChange={e => updateTarget('trades_in', e.target.value)}
                  className="w-full h-1 bg-surface-container-high rounded-full appearance-none outline-none accent-primary"
                />
              </div>
            </div>

            {/* Topology Controls */}
            <div className="flex flex-col gap-4 pt-6 border-t border-outline-variant/20">
              <h3 className="font-label-caps text-xs text-on-surface-variant uppercase tracking-widest">Graph Topology</h3>
              <div className="grid grid-cols-2 gap-2">
                <button 
                  onClick={() => addNeighbor('safe')}
                  className="bg-surface-container-high border border-outline-variant/50 hover:border-tertiary/50 hover:bg-tertiary/10 text-on-surface font-data-mono py-4 rounded flex flex-col items-center justify-center gap-2 transition-all active:scale-95 group"
                >
                  <ShieldCheck className="text-tertiary group-hover:drop-shadow-[0_0_8px_rgba(97,246,185,0.8)]" size={20} />
                  <span className="text-[10px]">Add Safe Edge</span>
                </button>
                <button 
                  onClick={() => addNeighbor('hacker')}
                  className="bg-surface-container-high border border-outline-variant/50 hover:border-error/50 hover:bg-error/10 text-on-surface font-data-mono py-4 rounded flex flex-col items-center justify-center gap-2 transition-all active:scale-95 group"
                >
                  <ShieldAlert className="text-error group-hover:drop-shadow-[0_0_8px_rgba(255,180,171,0.8)]" size={20} />
                  <span className="text-[10px]">Add Risk Edge</span>
                </button>
              </div>
              {neighbors.length > 0 && (
                <button 
                  onClick={clearNeighbors}
                  className="w-full py-2 text-xs text-outline hover:text-on-surface transition-colors border border-outline-variant/30 rounded"
                >
                  Clear All Connections
                </button>
              )}
            </div>
          </div>
        </section>

        {/* Right Panel: Visuals */}
        <section className="flex-1 flex flex-col relative p-8 gap-8 overflow-y-auto">
          {/* Risk Gauge Card */}
          <div 
            className="glass-strong rounded-xl p-8 flex flex-col items-center justify-center relative z-10 border border-white/10 glow-red shrink-0 min-h-[250px]"
            style={{ borderColor: scoreColor }}
          >
            <div className="absolute top-4 left-4 flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full animate-pulse`} style={{ backgroundColor: scoreColor }}></div>
              <span className="font-label-caps text-xs tracking-widest uppercase" style={{ color: scoreColor }}>
                Live Fraud Risk Analysis
              </span>
            </div>

            <div className="flex flex-col items-center">
              <div className="relative flex items-center justify-center">
                <svg className="w-48 h-48 transform -rotate-90" viewBox="0 0 100 100">
                  <circle cx="50" cy="50" fill="none" r="45" stroke="rgba(255,255,255,0.05)" strokeDasharray="2 4" strokeWidth="4"></circle>
                  <circle 
                    className="transition-all duration-1000 ease-out"
                    cx="50" cy="50" fill="none" r="45" 
                    stroke={scoreColor}
                    strokeDasharray="283" 
                    strokeDashoffset={283 - (riskScore * 283)} 
                    strokeWidth="4"
                    style={{ filter: `drop-shadow(0 0 8px ${scoreColor}80)` }}
                  ></circle>
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center" style={{ color: scoreColor }}>
                  <span className="font-data-lg text-5xl font-bold tracking-tighter">
                    {(riskScore * 100).toFixed(1)}
                  </span>
                  <span className="font-data-mono text-sm opacity-80">%</span>
                </div>
              </div>
              <div className="mt-4 font-data-mono text-sm text-on-surface-variant flex items-center gap-2">
                <span>STATUS: {isHighRisk ? 'CRITICAL THREAT' : isMediumRisk ? 'SUSPICIOUS ACTIVITY' : 'SECURE NODE'}</span>
              </div>
            </div>
          </div>

          {/* Message Passing Visualization */}
          <div className="flex-1 glass rounded-xl flex flex-col relative z-10 border border-white/5 overflow-hidden min-h-[400px]">
            <div className="p-4 border-b border-outline-variant/20 flex justify-between items-center bg-surface-dim/50 backdrop-blur-md absolute top-0 left-0 w-full z-20">
              <h3 className="font-h2 text-sm text-on-surface flex items-center gap-2">
                <Activity size={18} className="text-primary" />
                GNN Message Passing Visualization
              </h3>
              <div className="flex gap-4">
                <span className="flex items-center gap-1 font-label-caps text-[9px] text-on-surface-variant">
                  <div className="w-2 h-2 rounded-full bg-tertiary"></div> Safe Trade
                </span>
                <span className="flex items-center gap-1 font-label-caps text-[9px] text-on-surface-variant">
                  <div className="w-2 h-2 rounded-full bg-error"></div> High Risk Trade
                </span>
              </div>
            </div>

            <div className="flex-1 relative w-full h-full bg-[radial-gradient(circle_at_center,_rgba(255,255,255,0.02)_1px,_transparent_1px)] bg-[size:20px_20px] flex items-center justify-center pt-16">
              <div className="topology-view">
                <div className="target-node-wrapper">
                  <div 
                    className="target-node" 
                    style={{ 
                      borderColor: scoreColor, 
                      boxShadow: `0 0 30px ${scoreColor}60`,
                      background: 'rgba(14, 20, 22, 0.8)'
                    }}
                  >
                    TARGET
                    <div className="absolute inset-0 rounded-full border border-primary/50 animate-ping opacity-20"></div>
                  </div>
                </div>
                
                <div className="neighbors-wrapper">
                  {neighbors.map((n, i) => {
                    const isHacker = n.sent > 1000000;
                    return (
                      <div 
                        key={i} 
                        className={`neighbor-node ${isHacker ? 'hacker-node' : 'safe-node'}`}
                        style={{
                          background: isHacker ? 'rgba(255, 180, 171, 0.1)' : 'rgba(97, 246, 185, 0.1)',
                          borderColor: isHacker ? '#ffb4ab' : '#61f6b9'
                        }}
                      >
                        {isHacker ? 'RISK' : 'SAFE'}
                      </div>
                    );
                  })}
                </div>
                
                {neighbors.length === 0 && (
                  <div className="empty-topology text-outline text-xs font-data-mono">
                    Node is isolated. Calculating baseline features...
                  </div>
                )}
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
