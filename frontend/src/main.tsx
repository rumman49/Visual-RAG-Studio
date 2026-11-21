import React from 'react'
import ReactDOM from 'react-dom/client'
import { AuthBoundary } from './AuthBoundary'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode><AuthBoundary /></React.StrictMode>,
)
