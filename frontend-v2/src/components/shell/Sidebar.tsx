"use client";

import * as React from "react";
import Link from "next/link";
import { Plus, MessageSquare, Settings, LogOut, PanelLeftClose, PanelLeftOpen, Search, Image, LayoutGrid, SearchCode, HelpCircle } from "lucide-react";
import { useAuth } from "@/lib/auth/AuthContext";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils/cn";

export function Sidebar() {
  const { user, status, signOut } = useAuth();
  const pathname = usePathname();
  const [isCollapsed, setIsCollapsed] = React.useState(false);

  const [history, setHistory] = React.useState<any[]>([]);

  React.useEffect(() => {
    // Dynamic import to avoid SSR mismatch since store uses localStorage
    import("@/lib/history/store").then(({ loadHistory }) => {
      const updateHistory = () => {
        setHistory(loadHistory(user?.sub));
      };
      
      updateHistory(); // Initial load

      window.addEventListener("hg-history-updated", updateHistory);
      return () => window.removeEventListener("hg-history-updated", updateHistory);
    });
  }, [user?.sub]);

  if (isCollapsed) {
    return (
      <div className="flex h-full w-[60px] flex-col items-center border-r border-border bg-surface-primary py-4 transition-all duration-300">
        <button
          onClick={() => setIsCollapsed(false)}
          className="mb-4 flex h-10 w-10 items-center justify-center rounded-lg text-text-secondary hover:bg-surface-hover hover:text-text-primary"
          aria-label="Expand sidebar"
        >
          <PanelLeftOpen className="h-5 w-5" />
        </button>
        <Link
          href="/app"
          className="flex h-10 w-10 items-center justify-center rounded-lg text-text-secondary hover:bg-surface-hover hover:text-text-primary"
          aria-label="New chat"
        >
          <Plus className="h-5 w-5" />
        </Link>
      </div>
    );
  }

  return (
    <div className="flex h-full w-[260px] shrink-0 flex-col bg-surface-primary transition-all duration-300">
      <div className="flex h-14 items-center justify-between px-3">
        <div className="flex flex-1 items-center gap-2">
          <button
            onClick={() => setIsCollapsed(true)}
            className="flex h-10 w-10 items-center justify-center rounded-lg text-text-secondary hover:bg-surface-hover hover:text-text-primary"
            aria-label="Collapse sidebar"
          >
            <PanelLeftClose className="h-5 w-5" />
          </button>
          <Link
            href="/app"
            className="flex h-10 w-10 items-center justify-center rounded-lg text-text-secondary hover:bg-surface-hover hover:text-text-primary"
          >
            <Plus className="h-5 w-5" />
          </Link>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-3 py-2">
        <div className="flex flex-col gap-1">
          <Link href="/app" className="flex items-center gap-3 rounded-lg px-3 py-2 text-[14px] text-text-primary hover:bg-surface-hover">
            <Plus className="h-4 w-4 shrink-0" />
            <span>New chat</span>
          </Link>
          <Link href="/history" className="flex items-center gap-3 rounded-lg px-3 py-2 text-[14px] text-text-primary hover:bg-surface-hover">
            <Search className="h-4 w-4 shrink-0" />
            <span>Search history</span>
          </Link>
          <button className="flex items-center gap-3 rounded-lg px-3 py-2 text-[14px] text-text-primary hover:bg-surface-hover">
            <Image className="h-4 w-4 shrink-0" />
            <span>Images</span>
          </button>
          <button className="flex items-center gap-3 rounded-lg px-3 py-2 text-[14px] text-text-primary hover:bg-surface-hover">
            <LayoutGrid className="h-4 w-4 shrink-0" />
            <span>Plugins</span>
          </button>
          <button className="flex items-center gap-3 rounded-lg px-3 py-2 text-[14px] text-text-primary hover:bg-surface-hover">
            <SearchCode className="h-4 w-4 shrink-0" />
            <span>Deep research</span>
          </button>
        </div>

        {status === "authenticated" && (
          <div className="mt-8">
            <div className="mb-2 px-3 text-[12px] font-semibold text-text-muted">Today</div>
            <div className="flex flex-col gap-1">
              {history.map((item) => (
                <Link
                  key={item.id}
                  href={`/app/verify/${item.id}`}
                  className={cn(
                    "flex items-center gap-2 rounded-lg px-3 py-2 text-[14px] text-text-secondary hover:bg-surface-hover hover:text-text-primary",
                    pathname === `/app/verify/${item.id}` && "bg-surface-hover text-text-primary"
                  )}
                >
                  <MessageSquare className="h-4 w-4 shrink-0" />
                  <span className="truncate">{item.query}</span>
                </Link>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="p-3">
        <div className="flex flex-col gap-1">
          <button className="flex items-center gap-3 rounded-lg px-3 py-2 text-[14px] text-text-primary hover:bg-surface-hover">
            <HelpCircle className="h-4 w-4 shrink-0" />
            <span>Help</span>
          </button>
          <button className="flex items-center gap-3 rounded-lg px-3 py-2 text-[14px] text-text-primary hover:bg-surface-hover">
            <Settings className="h-4 w-4 shrink-0" />
            <span>Settings</span>
          </button>
          <div className="my-1 h-px bg-border" />
          <div className="group flex items-center justify-between rounded-lg px-3 py-2 hover:bg-surface-hover">
            <div className="flex items-center gap-2 overflow-hidden">
              <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-surface-tertiary text-xs font-medium text-text-primary">
                {user?.name?.charAt(0).toUpperCase() || "U"}
              </div>
              <span className="truncate text-[14px] text-text-primary">
                {user?.name || user?.email || "User"}
              </span>
            </div>
            <button
              onClick={signOut}
              className="opacity-0 transition-opacity group-hover:opacity-100"
              aria-label="Sign out"
            >
              <LogOut className="h-4 w-4 text-text-secondary hover:text-text-primary" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
