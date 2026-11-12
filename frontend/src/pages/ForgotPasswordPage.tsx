import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Card, Button, Input } from '../components/ui';
import { apiForgotPassword, apiResetPassword } from '../api';
import { ShieldAlert, CheckCircle2, AlertCircle } from 'lucide-react';

export const ForgotPasswordPage: React.FC = () => {
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [resetToken, setResetToken] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [step, setStep] = useState<'request' | 'reset'>('request');
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleRequest = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const res = await apiForgotPassword(email);
      setMessage(res.message);
      if (res.reset_token) {
        setResetToken(res.reset_token);
      }
      setStep('reset');
    } catch (err: any) {
      setError(err.message || 'Error processing request.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const res = await apiResetPassword(resetToken, newPassword);
      alert('Password reset successful! Please sign in with your new password.');
      navigate('/login');
    } catch (err: any) {
      setError(err.message || 'Error resetting password.');
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
            }}
          >
            <ShieldAlert size={26} />
          </div>
          <h1 style={{ fontSize: '24px', fontWeight: 800, margin: '0 0 6px', color: '#F2F0E9' }}>
            Password Recovery
          </h1>
          <p style={{ fontSize: '13px', color: '#A3ACA5', margin: 0 }}>
            {step === 'request'
              ? 'Enter your registered email to request a reset token'
              : 'Enter your reset token and new password'}
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

          {message && (
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                padding: '10px 14px',
                borderRadius: '8px',
                backgroundColor: 'rgba(85, 167, 123, 0.15)',
                border: '1px solid #55A77B',
                color: '#55A77B',
                fontSize: '13px',
                marginBottom: '18px',
              }}
            >
              <CheckCircle2 size={16} />
              <span>{message}</span>
            </div>
          )}

          {step === 'request' ? (
            <form onSubmit={handleRequest} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div>
                <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: '#A3ACA5', marginBottom: '6px' }}>
                  REGISTERED EMAIL
                </label>
                <Input
                  type="email"
                  required
                  placeholder="user@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                />
              </div>

              <Button
                type="submit"
                variant="primary"
                disabled={loading}
                style={{ width: '100%', padding: '11px', marginTop: '4px' }}
              >
                {loading ? 'Generating Token...' : 'Send Reset Token'}
              </Button>
            </form>
          ) : (
            <form onSubmit={handleReset} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div>
                <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: '#A3ACA5', marginBottom: '6px' }}>
                  RESET TOKEN
                </label>
                <Input
                  type="text"
                  required
                  placeholder="RESET-..."
                  value={resetToken}
                  onChange={(e) => setResetToken(e.target.value)}
                />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: '#A3ACA5', marginBottom: '6px' }}>
                  NEW PASSWORD
                </label>
                <Input
                  type="password"
                  required
                  placeholder="••••••••••••"
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                />
              </div>

              <Button
                type="submit"
                variant="primary"
                disabled={loading}
                style={{ width: '100%', padding: '11px', marginTop: '4px' }}
              >
                {loading ? 'Updating Password...' : 'Reset Password'}
              </Button>
            </form>
          )}
        </Card>

        <div style={{ textAlign: 'center', marginTop: '18px', fontSize: '13px', color: '#A3ACA5' }}>
          Remembered password?{' '}
          <Link to="/login" style={{ color: '#F2A23A', fontWeight: 600 }}>
            Back to Sign In
          </Link>
        </div>
      </div>
    </div>
  );
};
