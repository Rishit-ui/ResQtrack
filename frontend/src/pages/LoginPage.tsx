import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { Card, Button, Input, Badge } from '../components/ui';
import { ShieldAlert, Lock, Mail, AlertCircle, ArrowRight } from 'lucide-react';

export const LoginPage: React.FC = () => {
  const { login } = useAuth();
  const navigate = useNavigate();

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const user = await login(email, password);
      if (user.role === 'ADMIN') {
        navigate('/admin/dashboard');
      } else if (user.role === 'RESPONDER') {
        navigate('/responder');
      } else {
        navigate('/user/dashboard');
      }
    } catch (err: any) {
      setError(err.message || 'Login failed. Please check credentials.');
    } finally {
      setLoading(false);
    }
  };

  const fillCredentials = (userEmail: string, userPass: string) => {
    setEmail(userEmail);
    setPassword(userPass);
    setError(null);
  };

  return (
    <div
      style={{
        minHeight: 'calc(100vh - 70px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '30px 20px',
        backgroundColor: '#090B0A',
      }}
    >
      <div style={{ width: '100%', maxWidth: '440px' }}>
        <div style={{ textAlign: 'center', marginBottom: '24px' }}>
          <div
            style={{
              width: '46px',
              height: '46px',
              borderRadius: '12px',
              backgroundColor: '#F2A23A',
              color: '#090B0A',
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              marginBottom: '14px',
              boxShadow: '0 0 20px rgba(242, 162, 58, 0.3)',
            }}
          >
            <ShieldAlert size={26} />
          </div>
          <h1 style={{ fontSize: '24px', fontWeight: 800, margin: '0 0 6px', color: '#F2F0E9' }}>
            Emergency Access Portal
          </h1>
          <p style={{ fontSize: '13px', color: '#A3ACA5', margin: 0 }}>
            Sign in to access your role-specific response dashboard
          </p>
        </div>

        <Card style={{ padding: '26px' }}>
          {error && (
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                padding: '10px 14px',
                borderRadius: '8px',
                backgroundColor: 'rgba(217, 87, 87, 0.15)',
                border: '1px solid #D95757',
                color: '#D95757',
                fontSize: '13px',
                marginBottom: '18px',
              }}
            >
              <AlertCircle size={16} />
              <span>{error}</span>
            </div>
          )}

          <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div>
              <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: '#A3ACA5', marginBottom: '6px' }}>
                EMAIL ADDRESS
              </label>
              <div style={{ position: 'relative' }}>
                <Input
                  type="email"
                  required
                  placeholder="operator@resqtrack.gov"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                />
              </div>
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                <label style={{ fontSize: '12px', fontWeight: 600, color: '#A3ACA5' }}>PASSWORD</label>
                <Link to="/forgot-password" style={{ fontSize: '12px', color: '#F2A23A' }}>
                  Forgot?
                </Link>
              </div>
              <Input
                type="password"
                required
                placeholder="••••••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>

            <Button
              type="submit"
              variant="primary"
              disabled={loading}
              style={{ width: '100%', padding: '11px', marginTop: '4px', fontSize: '14px' }}
            >
              {loading ? 'Authenticating...' : 'Sign In'} <ArrowRight size={16} />
            </Button>
          </form>

          {/* Quick Demo Logins */}
          <div style={{ marginTop: '24px', paddingTop: '18px', borderTop: '1px solid #303832' }}>
            <div style={{ fontSize: '11px', color: '#A3ACA5', fontWeight: 600, textTransform: 'uppercase', marginBottom: '10px', textAlign: 'center' }}>
              Quick Demo Accounts
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px' }}>
              <button
                type="button"
                onClick={() => fillCredentials('admin@resqtrack.gov', 'Admin@ResQTrack2026!')}
                style={{
                  backgroundColor: '#1D2420',
                  border: '1px solid #303832',
                  borderRadius: '6px',
                  padding: '8px 4px',
                  color: '#F2A23A',
                  fontSize: '11px',
                  fontWeight: 600,
                  cursor: 'pointer',
                  textAlign: 'center',
                }}
              >
                Admin
              </button>
              <button
                type="button"
                onClick={() => fillCredentials('responder@resqtrack.gov', 'Responder@2026!')}
                style={{
                  backgroundColor: '#1D2420',
                  border: '1px solid #303832',
                  borderRadius: '6px',
                  padding: '8px 4px',
                  color: '#55A77B',
                  fontSize: '11px',
                  fontWeight: 600,
                  cursor: 'pointer',
                  textAlign: 'center',
                }}
              >
                Responder
              </button>
              <button
                type="button"
                onClick={() => fillCredentials('citizen@resqtrack.org', 'Citizen@2026!')}
                style={{
                  backgroundColor: '#1D2420',
                  border: '1px solid #303832',
                  borderRadius: '6px',
                  padding: '8px 4px',
                  color: '#F2F0E9',
                  fontSize: '11px',
                  fontWeight: 600,
                  cursor: 'pointer',
                  textAlign: 'center',
                }}
              >
                Citizen
              </button>
            </div>
          </div>
        </Card>

        <div style={{ textAlign: 'center', marginTop: '18px', fontSize: '13px', color: '#A3ACA5' }}>
          New citizen user?{' '}
          <Link to="/register" style={{ color: '#F2A23A', fontWeight: 600 }}>
            Register an account
          </Link>
        </div>
      </div>
    </div>
  );
};
