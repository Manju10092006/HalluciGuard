import React, { useRef, useEffect, useState } from 'react'
import { ArrowDown } from 'lucide-react'
import { UserMessage } from './UserMessage'
import { HalluciGuardMessage } from './HalluciGuardMessage'
import { VerificationProgress } from '../loading/VerificationProgress'

export function Conversation({
  messages = [],
  isLoading = false,
  onOpenSource,
}) {
  const scrollAnchorRef = useRef(null)
  const containerRef = useRef(null)
  const [showScrollBottom, setShowScrollBottom] = useState(false)

  const scrollToBottom = (smooth = true) => {
    scrollAnchorRef.current?.scrollIntoView({
      behavior: smooth ? 'smooth' : 'auto',
    })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages, isLoading])

  const handleScroll = () => {
    if (!containerRef.current) return
    const { scrollTop, scrollHeight, clientHeight } = containerRef.current
    const isScrolledUp = scrollHeight - scrollTop - clientHeight > 180
    setShowScrollBottom(isScrolledUp)
  }

  return (
    <div
      className="conversation-scroll-area"
      ref={containerRef}
      onScroll={handleScroll}
    >
      <div className="conversation-column">
        {messages.map((msg) =>
          msg.sender === 'user' ? (
            <UserMessage key={msg.id} message={msg} />
          ) : (
            <HalluciGuardMessage
              key={msg.id}
              message={msg}
              onOpenSource={onOpenSource}
            />
          )
        )}

        {isLoading && <VerificationProgress />}

        <div ref={scrollAnchorRef} style={{ height: 1 }} />
      </div>

      {showScrollBottom && (
        <button
          type="button"
          className="scroll-to-bottom-btn"
          onClick={() => scrollToBottom(true)}
          aria-label="Scroll to newest message"
        >
          <ArrowDown size={16} />
        </button>
      )}

      <style>{`
        .conversation-scroll-area {
          flex: 1;
          overflow-y: auto;
          overflow-x: hidden;
          padding: 24px 20px 40px;
          display: flex;
          justify-content: center;
          position: relative;
        }

        .conversation-column {
          width: 100%;
          max-width: var(--conversation-max-width);
          display: flex;
          flex-direction: column;
        }

        .scroll-to-bottom-btn {
          position: fixed;
          bottom: 140px;
          left: 50%;
          transform: translateX(-50%);
          width: 36px;
          height: 36px;
          border-radius: 50%;
          background: var(--surface);
          border: 1px solid var(--border);
          box-shadow: var(--shadow-float);
          color: var(--text-secondary);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 15;
          transition: all 140ms ease;
        }

        .scroll-to-bottom-btn:hover {
          color: var(--accent);
          background: var(--surface-sunken);
          transform: translateX(-50%) translateY(-2px);
        }
      `}</style>
    </div>
  )
}
