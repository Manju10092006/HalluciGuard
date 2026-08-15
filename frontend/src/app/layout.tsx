import type { Metadata } from 'next';
import './globals.css';
import { AnalyticsModal, SettingsModal, AuditReportModal } from '@/components/modals';

export const metadata: Metadata = {
  title: 'HalluciGuard - AI Verification Operating System',
  description: 'Real-time multi-agent verification engine for AI responses. Visualize reasoning, claims, and ground truth evidence.',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="bg-zinc-950 text-zinc-100 antialiased font-sans">
        {children}
        <AnalyticsModal />
        <SettingsModal />
        <AuditReportModal />
      </body>
    </html>
  );
}
