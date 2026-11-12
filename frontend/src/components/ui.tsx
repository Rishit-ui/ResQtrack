import React from 'react';

export const Card: React.FC<{
  children: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
  onClick?: () => void;
}> = ({ children, className = '', style = {}, onClick }) => (
  <div
    onClick={onClick}
    className={className}
    style={{
      backgroundColor: '#151917',
      border: '1px solid #303832',
      borderRadius: '12px',
      padding: '16px',
      boxShadow: '0 4px 12px rgba(0, 0, 0, 0.4)',
      ...style,
    }}
  >
    {children}
  </div>
);

export const ElevatedCard: React.FC<{
  children: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
}> = ({ children, className = '', style = {} }) => (
  <div
    className={className}
    style={{
      backgroundColor: '#1D2420',
      border: '1px solid #303832',
      borderRadius: '12px',
      padding: '18px',
      ...style,
    }}
  >
    {children}
  </div>
);

export const Badge: React.FC<{
  children: React.ReactNode;
  variant?: 'amber' | 'success' | 'warning' | 'danger' | 'neutral';
  style?: React.CSSProperties;
  className?: string;
}> = ({ children, variant = 'neutral', style = {}, className = '' }) => {
  const styles: Record<string, { bg: string; text: string; border: string }> = {
    amber: { bg: 'rgba(242, 162, 58, 0.12)', text: '#F2A23A', border: 'rgba(242, 162, 58, 0.3)' },
    success: { bg: 'rgba(85, 167, 123, 0.12)', text: '#55A77B', border: 'rgba(85, 167, 123, 0.3)' },
    warning: { bg: 'rgba(231, 174, 75, 0.12)', text: '#E7AE4B', border: 'rgba(231, 174, 75, 0.3)' },
    danger: { bg: 'rgba(217, 87, 87, 0.15)', text: '#D95757', border: 'rgba(217, 87, 87, 0.35)' },
    neutral: { bg: 'rgba(163, 172, 165, 0.12)', text: '#A3ACA5', border: 'rgba(163, 172, 165, 0.25)' },
  };
  const current = styles[variant] || styles.neutral;

  return (
    <span
      className={className}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        padding: '2px 8px',
        borderRadius: '6px',
        fontSize: '11px',
        fontWeight: 600,
        textTransform: 'uppercase',
        letterSpacing: '0.5px',
        backgroundColor: current.bg,
        color: current.text,
        border: `1px solid ${current.border}`,
        ...style,
      }}
    >
      {children}
    </span>
  );
};

export const Button: React.FC<{
  children: React.ReactNode;
  variant?: 'primary' | 'secondary' | 'danger' | 'success' | 'outline';
  onClick?: (e?: any) => void;
  disabled?: boolean;
  type?: 'button' | 'submit' | 'reset';
  style?: React.CSSProperties;
  className?: string;
  title?: string;
}> = ({ children, variant = 'primary', onClick, disabled = false, type = 'button', style = {}, className = '', title }) => {
  const baseStyle: React.CSSProperties = {
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '8px',
    padding: '8px 16px',
    borderRadius: '8px',
    fontSize: '13px',
    fontWeight: 600,
    cursor: disabled ? 'not-allowed' : 'pointer',
    opacity: disabled ? 0.5 : 1,
    transition: 'all 0.15s ease',
    border: 'none',
  };

  const variants: Record<string, React.CSSProperties> = {
    primary: {
      backgroundColor: '#F2A23A',
      color: '#090B0A',
    },
    secondary: {
      backgroundColor: '#1D2420',
      color: '#F2F0E9',
      border: '1px solid #303832',
    },
    danger: {
      backgroundColor: '#D95757',
      color: '#FFFFFF',
    },
    success: {
      backgroundColor: '#55A77B',
      color: '#FFFFFF',
    },
    outline: {
      backgroundColor: 'transparent',
      color: '#F2F0E9',
      border: '1px solid #303832',
    },
  };

  return (
    <button
      type={type}
      disabled={disabled}
      onClick={onClick}
      className={className}
      title={title}
      style={{ ...baseStyle, ...variants[variant], ...style }}
    >
      {children}
    </button>
  );
};

export const Input: React.FC<React.InputHTMLAttributes<HTMLInputElement>> = (props) => (
  <input
    {...props}
    style={{
      width: '100%',
      backgroundColor: '#111512',
      border: '1px solid #303832',
      borderRadius: '8px',
      padding: '10px 12px',
      color: '#F2F0E9',
      fontSize: '13px',
      outline: 'none',
      boxSizing: 'border-box',
      ...props.style,
    }}
  />
);

export const Textarea: React.FC<React.TextareaHTMLAttributes<HTMLTextAreaElement>> = (props) => (
  <textarea
    {...props}
    style={{
      width: '100%',
      backgroundColor: '#111512',
      border: '1px solid #303832',
      borderRadius: '8px',
      padding: '10px 12px',
      color: '#F2F0E9',
      fontSize: '13px',
      outline: 'none',
      minHeight: '80px',
      boxSizing: 'border-box',
      ...props.style,
    }}
  />
);
