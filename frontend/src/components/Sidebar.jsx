import { NavLink, useLocation } from 'react-router-dom';
import { LayoutDashboard, Users, Network, ShieldAlert, BarChart3, Search, Zap } from 'lucide-react';
import './Sidebar.css';

const navItems = [
  { path: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { path: '/players', icon: Users, label: 'Players' },
  { path: '/graph', icon: Network, label: 'Network Graph' },
  { path: '/analytics', icon: BarChart3, label: 'Analytics' },
  { path: '/sandbox', icon: Zap, label: 'AI Sandbox' },
];

export default function Sidebar() {
  const location = useLocation();

  return (
    <aside className="w-64 h-full bg-surface-container-lowest border-r border-white/5 flex flex-col z-50 shadow-[10px_0_30px_rgba(0,0,0,0.5)]">
      {/* Brand Header */}
      <div className="px-6 py-8 flex flex-col gap-1 border-b border-white/5 bg-surface-dim/30">
        <span className="text-outline font-data-mono text-[9px] tracking-widest uppercase">Protocol v4.2_SIGMA</span>
        <div className="flex items-center gap-2">
          <ShieldAlert className="text-primary" size={20} />
          <span className="text-xl font-black text-on-surface tracking-tighter uppercase">AEGIS COMMAND</span>
        </div>
      </div>

      {/* Navigation Links */}
      <nav className="flex-1 overflow-y-auto py-6 flex flex-col gap-1">
        {navItems.map(item => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) =>
              `flex items-center gap-3 py-4 px-6 transition-all duration-200 group relative ${
                isActive 
                  ? 'bg-primary/10 text-primary border-r-4 border-primary' 
                  : 'text-outline hover:bg-white/[0.03] hover:text-on-surface'
              }`
            }
          >
            <item.icon size={18} className="group-hover:scale-110 transition-transform" />
            <span className="font-semibold text-xs uppercase tracking-widest">{item.label}</span>
            {location.pathname === item.path && (
              <div className="absolute inset-0 bg-gradient-to-r from-primary/5 to-transparent pointer-events-none" />
            )}
          </NavLink>
        ))}
      </nav>

      {/* Sidebar Footer */}
      <div className="p-6 border-t border-white/5 bg-surface-dim/20">
        <div className="flex justify-between items-center px-2 text-outline">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-tertiary animate-pulse"></div>
            <span className="text-[10px] font-data-mono uppercase text-tertiary">Engine_Active</span>
          </div>
          <span className="text-[10px] font-data-mono opacity-50">v1.0.4-LTS</span>
        </div>
      </div>
    </aside>
  );
}
