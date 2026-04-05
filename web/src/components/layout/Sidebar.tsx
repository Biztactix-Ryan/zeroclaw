import { NavLink, useLocation } from 'react-router-dom';
import { basePath } from '../../lib/basePath';
import {
  LayoutDashboard,
  MessageSquare,
  Wrench,
  Clock,
  Puzzle,
  Brain,
  Settings,
  DollarSign,
  Activity,
  Stethoscope,
  Monitor,
  Blocks,
} from 'lucide-react';
import { t } from '@/lib/i18n';

interface NavChild {
  to: string;
  labelKey: string;
}

interface NavItem {
  to: string;
  icon: typeof LayoutDashboard;
  labelKey: string;
  children?: NavChild[];
}

const navItems: NavItem[] = [
  { to: '/', icon: LayoutDashboard, labelKey: 'nav.dashboard' },
  { to: '/agent', icon: MessageSquare, labelKey: 'nav.agent' },
  { to: '/tools', icon: Wrench, labelKey: 'nav.tools' },
  { to: '/cron', icon: Clock, labelKey: 'nav.cron' },
  {
    to: '/integrations',
    icon: Puzzle,
    labelKey: 'nav.integrations',
    children: [
      { to: '/plugins', labelKey: 'nav.plugins' },
    ],
  },
  { to: '/memory', icon: Brain, labelKey: 'nav.memory' },
  { to: '/config', icon: Settings, labelKey: 'nav.config' },
  { to: '/cost', icon: DollarSign, labelKey: 'nav.cost' },
  { to: '/logs', icon: Activity, labelKey: 'nav.logs' },
  { to: '/doctor', icon: Stethoscope, labelKey: 'nav.doctor' },
  { to: '/canvas', icon: Monitor, labelKey: 'nav.canvas' },
];

interface SidebarProps {
  open: boolean;
  onClose: () => void;
}

export default function Sidebar({ open, onClose }: SidebarProps) {
  const location = useLocation();

  return (
    <>
      {/* Backdrop — mobile only, visible when sidebar is open */}
      {open && (
        <div
          className="md:hidden fixed inset-0 z-40 bg-black/60 backdrop-blur-sm transition-opacity"
          onClick={onClose}
          onKeyDown={(e) => { if (e.key === 'Escape') onClose(); }}
          role="button"
          tabIndex={-1}
          aria-label="Close menu"
        />
      )}

      <aside
        className={[
          'fixed top-0 left-0 h-screen w-60 flex flex-col border-r z-50',
          // Mobile: slide in/out with transition
          'max-md:-translate-x-full max-md:transition-transform max-md:duration-200 max-md:ease-out',
          open ? 'max-md:translate-x-0' : '',
        ].join(' ')}
        style={{ background: 'var(--pc-bg-base)', borderColor: 'var(--pc-border)' }}
      >
        {/* Logo / Title */}
        <div className="flex items-center gap-3 px-4 py-4 border-b h-14" style={{ borderColor: 'var(--pc-border)' }}>
          <div className="relative shrink-0">
            <div className="absolute -inset-1.5 rounded-xl" style={{ background: 'linear-gradient(135deg, rgba(var(--pc-accent-rgb), 0.15), rgba(var(--pc-accent-rgb), 0.05))' }} />
            <img
              src={`${basePath}/_app/zeroclaw-trans.png`}
              alt="ZeroClaw"
              className="relative h-9 w-9 rounded-xl object-cover"
              onError={(e) => {
                const img = e.currentTarget;
                img.style.display = 'none';
              }}
            />
          </div>
          <span className="text-sm font-semibold tracking-wide" style={{ color: 'var(--pc-text-primary)' }}>
            ZeroClaw
          </span>
        </div>

        {/* Navigation */}
        <nav className="flex-1 overflow-y-auto py-4 px-3 space-y-1">
          {navItems.map(({ to, icon: Icon, labelKey, children }, idx) => (
            <div key={to}>
              <NavLink
                to={to}
                end={to === '/' || !!children}
                onClick={onClose}
                className={({ isActive }) =>
                  [
                    'flex items-center gap-3 px-3 py-2.5 rounded-2xl text-sm font-medium transition-all group',
                    isActive
                      ? 'text-[var(--pc-accent-light)]'
                      : 'text-[var(--pc-text-muted)] hover:text-[var(--pc-text-secondary)] hover:bg-[var(--pc-hover)]',
                  ].join(' ')
                }
                style={({ isActive }) => ({
                  animationDelay: `${idx * 40}ms`,
                  ...(isActive ? {
                    background: 'var(--pc-accent-glow)',
                    border: '1px solid var(--pc-accent-dim)',
                  } : {}),
                })}
              >
                {({ isActive }) => (
                  <>
                    <Icon className={`h-5 w-5 flex-shrink-0 transition-colors ${isActive ? 'text-[var(--pc-accent)]' : 'group-hover:text-[var(--pc-accent)]'}`} />
                    <span>{t(labelKey)}</span>
                  </>
                )}
              </NavLink>
              {children && (
                <div className="ml-6 mt-1 space-y-1">
                  {children.map((child) => {
                    const isChildActive = location.pathname === child.to || location.pathname.startsWith(child.to + '/');
                    return (
                      <NavLink
                        key={child.to}
                        to={child.to}
                        onClick={onClose}
                        className={[
                          'flex items-center gap-2 pl-5 pr-3 py-2 rounded-xl text-sm transition-all group',
                          isChildActive
                            ? 'text-[var(--pc-accent-light)]'
                            : 'text-[var(--pc-text-muted)] hover:text-[var(--pc-text-secondary)] hover:bg-[var(--pc-hover)]',
                        ].join(' ')}
                        style={isChildActive ? {
                          background: 'var(--pc-accent-glow)',
                          border: '1px solid var(--pc-accent-dim)',
                        } : {}}
                      >
                        <Blocks className={`h-4 w-4 flex-shrink-0 transition-colors ${isChildActive ? 'text-[var(--pc-accent)]' : 'group-hover:text-[var(--pc-accent)]'}`} />
                        <span>{t(child.labelKey)}</span>
                      </NavLink>
                    );
                  })}
                </div>
              )}
            </div>
          ))}
        </nav>

        {/* Footer */}
        <div className="px-5 py-4 border-t text-[10px] uppercase tracking-wider" style={{ borderColor: 'var(--pc-border)', color: 'var(--pc-text-faint)' }}>
          ZeroClaw Runtime
        </div>
      </aside>
    </>
  );
}
