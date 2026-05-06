import { NavLink, useLocation } from 'react-router-dom';
import { LayoutDashboard, Users, Network, ShieldAlert, BarChart3, Zap } from 'lucide-react';
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
    <aside className="sidebar-panel">
      <div className="sidebar-brand">
        <span className="sidebar-kicker">Protocol v4.2_SIGMA</span>
        <div className="sidebar-title-row">
          <span className="sidebar-logo">
            <ShieldAlert size={20} />
          </span>
          <span className="sidebar-title">AEGIS COMMAND</span>
        </div>
      </div>

      <nav className="sidebar-nav" aria-label="Primary navigation">
        {navItems.map(item => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) =>
              `sidebar-link ${isActive ? 'sidebar-link-active' : ''}`
            }
          >
            <item.icon size={18} />
            <span>{item.label}</span>
            {location.pathname === item.path && (
              <span className="sidebar-active-glow" aria-hidden="true" />
            )}
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className="engine-status">
          <span className="engine-dot" aria-hidden="true" />
          <span>Engine_Active</span>
        </div>
        <span className="sidebar-version">v1.0.4-LTS</span>
      </div>
    </aside>
  );
}
