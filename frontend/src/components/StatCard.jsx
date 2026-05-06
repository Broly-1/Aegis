import './StatCard.css';

export default function StatCard({ icon: Icon, label, value, subValue, color = 'primary', delay = 0 }) {
  const colorClass = `stat-card-${color}`;

  return (
    <div
      className={`stat-card glass-panel stat-card-accent ${colorClass}`}
      style={{ animationDelay: `${delay}ms` }}
    >
      <div className="stat-card-header">
        <div className="stat-card-icon">
          {Icon && <Icon size={20} />}
        </div>
        <span className="stat-card-label">{label}</span>
      </div>
      <div className="stat-card-value">{value}</div>
      {subValue && <div className="stat-card-sub">{subValue}</div>}
    </div>
  );
}
