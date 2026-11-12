import React from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { useRealtime } from '../context/RealtimeContext';
import { Badge, Button } from './ui';
import { ShieldAlert, Activity, LogOut, Radio, UserCheck, Menu, X } from 'lucide-react';

export const Navbar: React.FC = () => {
  const { user, isAuthenticated, logout } = useAuth();
  const { status: wsStatus } = useRealtime();
  const navigate = useNavigate();
  const location = useLocation();
  const [mobileMenuOpen, setMobileMenuOpen] = React.useState(false);

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const getDashboardPath = () => {
    if (!user) return '/login';
    if (user.role === 'ADMIN') return '/admin/dashboard';
    if (user.role === 'RESPONDER') return '/responder';
    return '/user/dashboard';
  };

  return (
    <header
      style={{
        backgroundColor: '#111512',
        borderBottom: '1px solid #303832',
        position: 'sticky',
        top: 0,
        zIndex: 50,
      }}
    >
      <div
        style={{
          maxWidth: '1440px',
          margin: '0 auto',
          padding: '12px 20px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        {/* Brand */}
        <Link to="/" style={{ display: 'flex', alignItems: 'center', gap: '10px', minWidth: 0, flexShrink: 1 }}>
          <div
            style={{
              width: '36px',
              height: '36px',
              minWidth: '36px',
              borderRadius: '10px',
              backgroundColor: '#F2A23A',
              color: '#090B0A',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontWeight: 800,
              boxShadow: '0 0 15px rgba(242, 162, 58, 0.3)',
            }}
          >
            <ShieldAlert size={20} />
          </div>
          <div style={{ minWidth: 0 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
              <span style={{ fontSize: '17px', fontWeight: 800, letterSpacing: '-0.3px', color: '#F2F0E9' }}>
                ResQ<span style={{ color: '#F2A23A' }}>Track</span>
              </span>
              <Badge variant="amber" className="desktop-nav" style={{ fontSize: '9px', padding: '1px 5px' }}>
                AI EMERGENCY
              </Badge>
            </div>
            <div className="brand-subtitle" style={{ fontSize: '11px', color: '#A3ACA5', letterSpacing: '0.2px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
              Autonomous Incident Detection & Response
            </div>
          </div>
        </Link>

        {/* Desktop Nav Links */}
        <nav
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '20px',
          }}
          className="desktop-nav"
        >
          <Link
            to="/"
            style={{
              fontSize: '13px',
              fontWeight: 500,
              color: location.pathname === '/' ? '#F2A23A' : '#A3ACA5',
              transition: 'color 0.15s ease',
            }}
          >
            Home
          </Link>

          {isAuthenticated ? (
            <>
              {user?.role === 'ADMIN' && (
                <Link
                  to="/admin/dashboard"
                  style={{
                    fontSize: '13px',
                    fontWeight: 500,
                    color: location.pathname.startsWith('/admin') ? '#F2A23A' : '#A3ACA5',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                  }}
                >
                  <Activity size={15} /> Command Centre
                </Link>
              )}

              {user?.role === 'RESPONDER' && (
                <Link
                  to="/responder"
                  style={{
                    fontSize: '13px',
                    fontWeight: 500,
                    color: location.pathname.startsWith('/responder') ? '#F2A23A' : '#A3ACA5',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                  }}
                >
                  <Radio size={15} /> Crew Portal
                </Link>
              )}

              {user?.role === 'CITIZEN' && (
                <Link
                  to="/user/dashboard"
                  style={{
                    fontSize: '13px',
                    fontWeight: 500,
                    color: location.pathname.startsWith('/user') ? '#F2A23A' : '#A3ACA5',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                  }}
                >
                  <UserCheck size={15} /> Citizen Portal
                </Link>
              )}
            </>
          ) : (
            <>
              <a href="#how-it-works" style={{ fontSize: '13px', color: '#A3ACA5' }}>
                How It Works
              </a>
              <a href="#features" style={{ fontSize: '13px', color: '#A3ACA5' }}>
                Features
              </a>
              <a href="#tech" style={{ fontSize: '13px', color: '#A3ACA5' }}>
                Tech
              </a>
            </>
          )}
        </nav>

        {/* Right Action & Status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          {/* Realtime Dot */}
          <div
            title={`Real-time WebSocket: ${wsStatus}`}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              backgroundColor: '#151917',
              border: '1px solid #303832',
              borderRadius: '20px',
              padding: '4px 10px',
              fontSize: '11px',
              color: '#A3ACA5',
            }}
          >
            <div
              style={{
                width: '8px',
                height: '8px',
                borderRadius: '50%',
                backgroundColor:
                  wsStatus === 'connected' ? '#55A77B' : wsStatus === 'connecting' ? '#E7AE4B' : '#D95757',
                boxShadow:
                  wsStatus === 'connected'
                    ? '0 0 8px #55A77B'
                    : wsStatus === 'connecting'
                    ? '0 0 8px #E7AE4B'
                    : 'none',
              }}
            />
            <span style={{ textTransform: 'uppercase', fontSize: '10px', fontWeight: 600 }}>
              {wsStatus === 'connected' ? 'LIVE' : wsStatus}
            </span>
          </div>

          {isAuthenticated ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <Link to={getDashboardPath()}>
                <div
                  style={{
                    backgroundColor: '#1D2420',
                    border: '1px solid #303832',
                    borderRadius: '8px',
                    padding: '6px 12px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                  }}
                >
                  <div style={{ textAlign: 'right' }}>
                    <div style={{ fontSize: '12px', fontWeight: 600, color: '#F2F0E9', lineHeight: 1.2 }}>
                      {user?.full_name}
                    </div>
                    <div style={{ fontSize: '10px', color: '#F2A23A', fontWeight: 600, textTransform: 'uppercase' }}>
                      {user?.role}
                    </div>
                  </div>
                </div>
              </Link>

              <Button
                variant="outline"
                onClick={handleLogout}
                style={{ padding: '6px 10px' }}
                title="Log out"
              >
                <LogOut size={15} />
              </Button>
            </div>
          ) : (
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Button variant="outline" onClick={() => navigate('/login')}>
                Sign In
              </Button>
              <Button variant="primary" onClick={() => navigate('/register')}>
                Register
              </Button>
            </div>
          )}

          {/* Mobile menu button */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            style={{
              display: 'none',
              background: 'none',
              border: 'none',
              color: '#F2F0E9',
              cursor: 'pointer',
              padding: '6px',
            }}
            className="mobile-toggle"
          >
            {mobileMenuOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>
      </div>
    </header>
  );
};
