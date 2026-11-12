import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { Card, Button, Input } from '../components/ui';
import { ShieldAlert, AlertCircle, ArrowRight } from 'lucide-react';

export const RegisterPage: React.FC = () => {
  const { register } = useAuth();
  const navigate = useNavigate();

  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [phone, setPhone] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      await register(email, password, fullName, phone);
      navigate('/user/dashboard');
    } catch (err: any) {
      setError(err.message || 'Registration failed.');
    } finally {
      setLoading(false);
    }
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
            Citizen Registration
          </h1>
          <p style={{ fontSize: '13px', color: '#A3ACA5', margin: 0 }}>
            Create an account to report road incidents and track rescue status
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
                FULL NAME
              </label>
              <Input
                type="text"
                required
                placeholder="Rishit Citizen"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
              />
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: '#A3ACA5', marginBottom: '6px' }}>
                EMAIL ADDRESS
              </label>
              <Input
                type="email"
                required
                placeholder="citizen@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: '#A3ACA5', marginBottom: '6px' }}>
                PHONE NUMBER (OPTIONAL)
              </label>
              <Input
                type="tel"
                placeholder="+91-98765-43210"
                value={phone}
                onChange={(e) => setPhone(e.target.value)}
              />
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: '#A3ACA5', marginBottom: '6px' }}>
                PASSWORD (MIN 6 CHARACTERS)
              </label>
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
              {loading ? 'Creating Account...' : 'Complete Registration'} <ArrowRight size={16} />
            </Button>
          </form>
        </Card>

        <div style={{ textAlign: 'center', marginTop: '18px', fontSize: '13px', color: '#A3ACA5' }}>
          Already have an account?{' '}
          <Link to="/login" style={{ color: '#F2A23A', fontWeight: 600 }}>
            Sign In
          </Link>
        </div>
      </div>
    </div>
  );
};
