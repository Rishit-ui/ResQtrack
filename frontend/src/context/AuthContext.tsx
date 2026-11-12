import React, { createContext, useContext, useState, useEffect } from 'react';
import { User } from '../types';
import { getToken, getStoredUser, clearToken, apiGetMe, apiLogin, apiRegister } from '../api';

interface AuthContextType {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (email: string, pass: string) => Promise<User>;
  register: (email: string, pass: string, name: string, phone?: string) => Promise<User>;
  logout: () => void;
  refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(getStoredUser());
  const [token, setTokenState] = useState<string | null>(getToken());
  const [isLoading, setIsLoading] = useState<boolean>(true);

  useEffect(() => {
    async function loadUser() {
      const storedToken = getToken();
      if (storedToken) {
        try {
          const freshUser = await apiGetMe();
          setUser(freshUser);
        } catch {
          clearToken();
          setUser(null);
          setTokenState(null);
        }
      }
      setIsLoading(false);
    }
    loadUser();
  }, []);

  const login = async (email: string, pass: string) => {
    const res = await apiLogin(email, pass);
    setUser(res.user);
    setTokenState(res.access_token);
    return res.user;
  };

  const register = async (email: string, pass: string, name: string, phone?: string) => {
    const res = await apiRegister(email, pass, name, phone);
    setUser(res.user);
    setTokenState(res.access_token);
    return res.user;
  };

  const logout = () => {
    clearToken();
    setUser(null);
    setTokenState(null);
  };

  const refreshUser = async () => {
    if (token) {
      try {
        const u = await apiGetMe();
        setUser(u);
      } catch {
        // ignore
      }
    }
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        isAuthenticated: !!user,
        isLoading,
        login,
        register,
        logout,
        refreshUser,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within an AuthProvider');
  return ctx;
};
