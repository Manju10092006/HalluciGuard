import React from 'react'
import { AuthDialog } from '../auth/AuthDialog'

// Compatibility wrapper: authentication is handled only by the server-backed provider.
export function AuthModal({ isOpen, onClose }) {
  return <AuthDialog open={isOpen} onClose={onClose} />
}
