'use client';

import React from 'react';
import { WorkspaceView } from '@/components/views/WorkspaceView';
import { useRouter } from 'next/navigation';

export default function ChatPage() {
  const router = useRouter();
  return <WorkspaceView onBackToLanding={() => router.push('/')} />;
}
