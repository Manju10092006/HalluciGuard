import React from 'react'
import { createRoot } from 'react-dom/client'
import { HalluciGuardShell } from './components/shell/HalluciGuardShell'
import './index.css'

const rootElement = document.getElementById('root')
if (rootElement) {
  createRoot(rootElement).render(
    <React.StrictMode>
      <HalluciGuardShell />
    </React.StrictMode>
  )
}
