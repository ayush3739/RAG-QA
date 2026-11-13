import React, { useState, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { 
  ShieldCheck, Search, Users, Crown, Check, X, Shield, RefreshCw, 
  ArrowUpRight, Clock, Mail, Copy, CheckCircle2, AlertCircle, FileText, Sparkles,
  ChevronLeft, ChevronRight
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { toast } from "sonner";
import { api } from "../lib/api";
import { cn } from "../lib/utils";
import { useStore } from "../store/useStore";

interface AdminUser {
  id: number;
  name: string;
  email: string;
  role: string;
  is_verified: boolean;
  created_at: string;
  document_count: number;
  query_count: number;
  has_pending_upgrade: boolean;
}

interface UpgradeRequest {
  id: number;
  user_id: number;
  user_name: string;
  user_email: string;
  current_role: string;
  requested_role: string;
  status: string;
  reason?: string;
  created_at: string;
}

interface AdminStats {
  total_users: number;
  pro_users: number;
  admin_users: number;
  pending_requests: number;
}

const PAGE_SIZE = 50;

export default function AdminView() {
  const queryClient = useQueryClient();
  const currentUser = useStore((s) => s.user);
  const [search, setSearch] = useState("");
  const [roleFilter, setRoleFilter] = useState<string>("all");
  const [page, setPage] = useState<number>(1);
  const [activeTab, setActiveTab] = useState<"users" | "requests">("users");
  const [copiedEmail, setCopiedEmail] = useState<string | null>(null);

  // Reset pagination to first page whenever search or filter changes
  useEffect(() => {
    setPage(1);
  }, [search, roleFilter]);

  // 1. Fetch Global Stats (Accurate counts regardless of table filters)
  const { data: statsData, refetch: refetchStats } = useQuery({
    queryKey: ["adminStats"],
    queryFn: async () => {
      const res = await api.getAdminStats();
      return res as AdminStats;
    },
    refetchOnWindowFocus: false,
  });

  // 2. Fetch Paged Users (Up to 50 users per page)
  const { data: usersData, isLoading: isLoadingUsers, refetch: refetchUsers } = useQuery({
    queryKey: ["adminUsers", search, roleFilter, page],
    queryFn: async () => {
      const res = await api.getAdminUsers({
        search: search.trim() || undefined,
        role: roleFilter === "all" ? undefined : roleFilter,
        limit: PAGE_SIZE,
        skip: (page - 1) * PAGE_SIZE,
      });
      return res as { users: AdminUser[]; total: number; limit: number; skip: number };
    },
    refetchOnWindowFocus: false,
  });

  // 3. Fetch Upgrade Requests
  const { data: requestsData, isLoading: isLoadingRequests, refetch: refetchRequests } = useQuery({
    queryKey: ["adminUpgradeRequests"],
    queryFn: async () => {
      const res = await api.getAdminUpgradeRequests("pending");
      return res as UpgradeRequest[];
    },
    refetchOnWindowFocus: false,
  });

  // 4. Role Change Mutation
  const updateRoleMutation = useMutation({
    mutationFn: async ({ userId, role }: { userId: number; role: "user" | "pro" | "admin" }) => {
      return await api.updateUserRole(userId, role);
    },
    onSuccess: (data, variables) => {
      toast.success(`User role updated to ${variables.role.toUpperCase()}`);
      queryClient.invalidateQueries({ queryKey: ["adminUsers"] });
      queryClient.invalidateQueries({ queryKey: ["adminStats"] });
      queryClient.invalidateQueries({ queryKey: ["adminUpgradeRequests"] });
    },
    onError: (err: any) => {
      toast.error(err.message || "Failed to update role");
    },
  });

  // 5. Approve Upgrade Mutation
  const approveMutation = useMutation({
    mutationFn: async (requestId: number) => {
      return await api.approveUpgradeRequest(requestId);
    },
    onSuccess: () => {
      toast.success("Upgrade request approved! User is now Pro.");
      queryClient.invalidateQueries({ queryKey: ["adminUsers"] });
      queryClient.invalidateQueries({ queryKey: ["adminStats"] });
      queryClient.invalidateQueries({ queryKey: ["adminUpgradeRequests"] });
    },
    onError: (err: any) => {
      toast.error(err.message || "Failed to approve request");
    },
  });

  // 6. Reject Upgrade Mutation
  const rejectMutation = useMutation({
    mutationFn: async (requestId: number) => {
      return await api.rejectUpgradeRequest(requestId);
    },
    onSuccess: () => {
      toast.info("Upgrade request rejected.");
      queryClient.invalidateQueries({ queryKey: ["adminStats"] });
      queryClient.invalidateQueries({ queryKey: ["adminUpgradeRequests"] });
    },
    onError: (err: any) => {
      toast.error(err.message || "Failed to reject request");
    },
  });

  const handleCopyEmail = (email: string) => {
    navigator.clipboard.writeText(email);
    setCopiedEmail(email);
    setTimeout(() => setCopiedEmail(null), 2000);
    toast.success("Email copied to clipboard");
  };

  const users = usersData?.users || [];
  const totalUsers = usersData?.total || 0;
  const requests = requestsData || [];

  const totalAccounts = statsData?.total_users ?? totalUsers;
  const proCount = statsData?.pro_users ?? users.filter((u) => u.role.toLowerCase() === "pro").length;
  const adminCount = statsData?.admin_users ?? users.filter((u) => u.role.toLowerCase() === "admin").length;
  const pendingCount = statsData?.pending_requests ?? requests.length;
  const totalPages = Math.max(1, Math.ceil(totalUsers / PAGE_SIZE));

  return (
    <div className="flex-1 flex flex-col h-full bg-background overflow-y-auto p-6 md:p-8 space-y-6">
      {/* Top Banner / Breadcrumb */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-border/60 pb-6">
        <div>
          <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full bg-primary/10 border border-primary/20 text-primary text-[11px] font-semibold tracking-wide uppercase mb-2">
            <ShieldCheck className="w-3.5 h-3.5" />
            Internal Administration Portal
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-foreground">
            User Access & Role Management
          </h1>
          <p className="text-xs text-muted-foreground mt-1">
            Search team members by email, review Pro access upgrade requests, and assign permissions.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => {
              refetchStats();
              refetchUsers();
              refetchRequests();
              toast.info("Refreshed user directory");
            }}
            className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-surface border border-border text-xs font-semibold text-foreground hover:bg-surface-container transition-colors shadow-sm cursor-pointer"
          >
            <RefreshCw className="w-3.5 h-3.5 text-muted-foreground" />
            Sync
          </button>
        </div>
      </div>

      {/* KPI Stats Strip */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="p-4 rounded-xl bg-surface border border-border shadow-sm">
          <div className="flex items-center justify-between text-muted-foreground mb-2">
            <span className="text-xs font-semibold">Total Accounts</span>
            <Users className="w-4 h-4" />
          </div>
          <div className="text-2xl font-bold text-foreground tabular-nums">{totalAccounts}</div>
          <div className="text-[11px] text-muted-foreground mt-1">Registered members</div>
        </div>

        <div className="p-4 rounded-xl bg-surface border border-border shadow-sm">
          <div className="flex items-center justify-between text-muted-foreground mb-2">
            <span className="text-xs font-semibold">Pro Tier Users</span>
            <Crown className="w-4 h-4 text-amber-500 dark:text-amber-400 fill-amber-500/20" />
          </div>
          <div className="text-2xl font-bold text-amber-600 dark:text-amber-400 tabular-nums">{proCount}</div>
          <div className="text-[11px] text-muted-foreground mt-1">500 queries/day & 50 slots</div>
        </div>

        <div className="p-4 rounded-xl bg-surface border border-border shadow-sm">
          <div className="flex items-center justify-between text-muted-foreground mb-2">
            <span className="text-xs font-semibold">Administrators</span>
            <Shield className="w-4 h-4 text-primary" />
          </div>
          <div className="text-2xl font-bold text-primary tabular-nums">{adminCount}</div>
          <div className="text-[11px] text-muted-foreground mt-1">Unmetered access</div>
        </div>

        <div className={cn(
          "p-4 rounded-xl border shadow-sm transition-colors",
          pendingCount > 0 
            ? "bg-amber-500/10 border-amber-500/30" 
            : "bg-surface border border-border"
        )}>
          <div className="flex items-center justify-between mb-2">
            <span className={cn("text-xs font-semibold", pendingCount > 0 ? "text-amber-400" : "text-muted-foreground")}>
              Pending Pro Requests
            </span>
            <div className="flex items-center gap-1.5">
              {pendingCount > 0 && <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />}
              <Clock className={cn("w-4 h-4", pendingCount > 0 ? "text-amber-400" : "text-muted-foreground")} />
            </div>
          </div>
          <div className={cn("text-2xl font-bold tabular-nums", pendingCount > 0 ? "text-amber-400" : "text-foreground")}>
            {pendingCount}
          </div>
          <div className="text-[11px] text-muted-foreground mt-1">
            {pendingCount > 0 ? "Needs admin approval" : "All requests resolved"}
          </div>
        </div>
      </div>

      {/* Tabs Switcher */}
      <div className="flex items-center gap-2 border-b border-border/60 pb-2">
        <button
          onClick={() => setActiveTab("users")}
          className={cn(
            "flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition-colors cursor-pointer",
            activeTab === "users"
              ? "bg-primary text-on-primary shadow-sm"
              : "text-muted-foreground hover:bg-surface-container hover:text-foreground"
          )}
        >
          <Users className="w-3.5 h-3.5" />
          User Directory ({totalUsers})
        </button>

        <button
          onClick={() => setActiveTab("requests")}
          className={cn(
            "flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition-colors cursor-pointer relative",
            activeTab === "requests"
              ? "bg-primary text-on-primary shadow-sm"
              : "text-muted-foreground hover:bg-surface-container hover:text-foreground"
          )}
        >
          <Crown className="w-3.5 h-3.5" />
          Pending Upgrade Requests
          {pendingCount > 0 && (
            <span className="ml-1 px-1.5 py-0.5 rounded-full bg-amber-500 text-black text-[10px] font-bold">
              {pendingCount}
            </span>
          )}
        </button>
      </div>

      {/* TAB 1: USERS DIRECTORY */}
      {activeTab === "users" && (
        <div className="space-y-4">
          {/* Search & Filter Bar */}
          <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3">
            <div className="relative flex-1 max-w-md">
              <Search className="w-4 h-4 text-muted-foreground absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search by email address or name..."
                className="w-full text-xs bg-surface border border-border rounded-lg pl-9 pr-8 py-2.5 text-foreground placeholder-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary transition-colors"
              />
              {search && (
                <button
                  onClick={() => setSearch("")}
                  className="absolute right-2.5 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground p-0.5"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              )}
            </div>

            <div className="flex items-center gap-1.5 overflow-x-auto pb-1 sm:pb-0">
              {(["all", "user", "pro", "admin"] as const).map((role) => (
                <button
                  key={role}
                  onClick={() => setRoleFilter(role)}
                  className={cn(
                    "px-3 py-1.5 rounded-lg text-xs font-medium capitalize transition-colors cursor-pointer",
                    roleFilter === role
                      ? "bg-surface-container-high text-foreground border border-border font-semibold shadow-xs"
                      : "text-muted-foreground hover:text-foreground hover:bg-surface-container"
                  )}
                >
                  {role === "all" ? "All Roles" : role}
                </button>
              ))}
            </div>
          </div>

          {/* Users Table */}
          <div className="rounded-xl border border-border bg-surface overflow-hidden shadow-sm">
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="border-b border-border bg-surface-container/40 text-muted-foreground font-semibold">
                    <th className="py-3 px-4">User</th>
                    <th className="py-3 px-4">Current Role</th>
                    <th className="py-3 px-4">Activity</th>
                    <th className="py-3 px-4">Joined</th>
                    <th className="py-3 px-4 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/60">
                  {isLoadingUsers ? (
                    <tr>
                      <td colSpan={5} className="py-8 text-center text-muted-foreground">
                        <RefreshCw className="w-4 h-4 animate-spin mx-auto mb-2 text-primary" />
                        Loading user directory...
                      </td>
                    </tr>
                  ) : users.length === 0 ? (
                    <tr>
                      <td colSpan={5} className="py-12 text-center text-muted-foreground">
                        <Users className="w-6 h-6 mx-auto mb-2 opacity-40" />
                        No users found matching &ldquo;{search}&rdquo;
                      </td>
                    </tr>
                  ) : (
                    users.map((u) => {
                      const roleLower = u.role.toLowerCase();
                      const isSelf = currentUser?.email === u.email;

                      return (
                        <tr key={u.id} className="hover:bg-surface-container/30 transition-colors">
                          {/* User info */}
                          <td className="py-3.5 px-4">
                            <div className="flex items-center gap-3">
                              <div className="w-8 h-8 rounded-full bg-primary/10 border border-primary/20 flex items-center justify-center text-primary font-bold text-xs uppercase shrink-0">
                                {u.name ? u.name.charAt(0) : u.email.charAt(0)}
                              </div>
                              <div className="min-w-0">
                                <div className="flex items-center gap-1.5 font-semibold text-foreground truncate">
                                  <span>{u.name || "Nameless User"}</span>
                                  {isSelf && (
                                    <span className="text-[10px] px-1.5 py-0.2 rounded bg-primary/10 text-primary font-medium">
                                      You
                                    </span>
                                  )}
                                  {u.has_pending_upgrade && (
                                    <span className="text-[10px] px-1.5 py-0.2 rounded bg-amber-500/10 text-amber-700 dark:text-amber-400 border border-amber-500/20 font-medium">
                                      Requested Pro
                                    </span>
                                  )}
                                </div>
                                <div className="flex items-center gap-1.5 text-muted-foreground text-[11px] font-mono mt-0.5">
                                  <span className="truncate">{u.email}</span>
                                  <button
                                    onClick={() => handleCopyEmail(u.email)}
                                    title="Copy email"
                                    className="hover:text-foreground transition-colors cursor-pointer"
                                  >
                                    {copiedEmail === u.email ? (
                                      <Check className="w-3 h-3 text-emerald-400" />
                                    ) : (
                                      <Copy className="w-3 h-3" />
                                    )}
                                  </button>
                                </div>
                              </div>
                            </div>
                          </td>

                          {/* Role Badge */}
                          <td className="py-3.5 px-4">
                            <span
                              className={cn(
                                "inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-semibold border",
                                roleLower === "admin"
                                  ? "bg-purple-500/10 text-purple-400 border-purple-500/20"
                                  : roleLower === "pro"
                                  ? "bg-amber-500/15 text-amber-700 dark:text-amber-300 border-amber-500/30 font-bold shadow-[0_0_8px_rgba(245,158,11,0.2)]"
                                  : "bg-surface-container text-muted-foreground border-border"
                              )}
                            >
                              {roleLower === "admin" && <Shield className="w-3 h-3 text-purple-400" />}
                              {roleLower === "pro" && <Crown className="w-3 h-3 text-amber-400 fill-amber-400" />}
                              <span className="capitalize">{u.role}</span>
                            </span>
                          </td>

                          {/* Stats */}
                          <td className="py-3.5 px-4 text-muted-foreground text-[11px]">
                            <div className="flex items-center gap-3">
                              <span title="Uploaded documents" className="flex items-center gap-1">
                                <FileText className="w-3 h-3 text-muted-foreground/70" />
                                <span className="text-foreground font-medium">{u.document_count}</span> docs
                              </span>
                              <span title="Total queries run" className="flex items-center gap-1">
                                <Sparkles className="w-3 h-3 text-muted-foreground/70" />
                                <span className="text-foreground font-medium">{u.query_count}</span> queries
                              </span>
                            </div>
                          </td>

                          {/* Joined */}
                          <td className="py-3.5 px-4 text-muted-foreground text-[11px] whitespace-nowrap">
                            {new Date(u.created_at).toLocaleDateString()}
                          </td>

                          {/* Action Buttons */}
                          <td className="py-3.5 px-4 text-right">
                            <div className="flex items-center justify-end gap-1.5">
                              {isSelf ? (
                                <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-semibold text-purple-400 bg-purple-500/10 border border-purple-500/25">
                                  <Shield className="w-3 h-3" />
                                  Current Admin (You)
                                </span>
                              ) : (
                                <>
                                  {roleLower !== "pro" && (
                                    <button
                                      type="button"
                                      disabled={updateRoleMutation.isPending}
                                      onClick={() => updateRoleMutation.mutate({ userId: u.id, role: "pro" })}
                                      className="px-2.5 py-1 rounded-md text-[11px] font-semibold bg-amber-500/10 hover:bg-amber-500/20 text-amber-700 dark:text-amber-300 border border-amber-500/30 transition-colors cursor-pointer flex items-center gap-1 shadow-xs"
                                      title="Upgrade to Pro tier (500 queries, 50 docs)"
                                    >
                                      <Crown className="w-3 h-3 text-amber-500 dark:text-amber-400 fill-amber-500/20" />
                                      Make Pro
                                    </button>
                                  )}

                                  {roleLower !== "admin" && (
                                    <button
                                      type="button"
                                      disabled={updateRoleMutation.isPending}
                                      onClick={() => updateRoleMutation.mutate({ userId: u.id, role: "admin" })}
                                      className="px-2.5 py-1 rounded-md text-[11px] font-semibold bg-purple-500/10 hover:bg-purple-500/20 text-purple-400 border border-purple-500/30 transition-colors cursor-pointer flex items-center gap-1 shadow-xs"
                                      title="Promote to Enterprise Administrator"
                                    >
                                      <Shield className="w-3 h-3" />
                                      Make Admin
                                    </button>
                                  )}

                                  {roleLower !== "user" && (
                                    <button
                                      type="button"
                                      disabled={updateRoleMutation.isPending}
                                      onClick={() => updateRoleMutation.mutate({ userId: u.id, role: "user" })}
                                      className="px-2 py-1 rounded-md text-[11px] font-medium text-muted-foreground hover:bg-surface-container transition-colors cursor-pointer"
                                      title="Demote to Standard Free User"
                                    >
                                      Demote to Free
                                    </button>
                                  )}
                                </>
                              )}
                            </div>
                          </td>
                        </tr>
                      );
                    })
                  )}
                </tbody>
              </table>
            </div>

            {/* Pagination Controls */}
            {totalUsers > 0 && (
              <div className="flex flex-col sm:flex-row items-center justify-between gap-3 px-4 py-3 border-t border-border bg-surface-container/20 text-xs">
                <div className="text-muted-foreground text-[11px]">
                  Showing <strong className="text-foreground font-semibold">{(page - 1) * PAGE_SIZE + 1}</strong> to{" "}
                  <strong className="text-foreground font-semibold">{Math.min(page * PAGE_SIZE, totalUsers)}</strong> of{" "}
                  <strong className="text-foreground font-semibold">{totalUsers}</strong> accounts
                </div>

                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    disabled={page <= 1 || isLoadingUsers}
                    onClick={() => setPage((prev) => Math.max(1, prev - 1))}
                    className="px-2.5 py-1.5 rounded-lg border border-border bg-surface hover:bg-surface-container text-foreground disabled:opacity-40 disabled:pointer-events-none transition-colors flex items-center gap-1 cursor-pointer font-medium"
                  >
                    <ChevronLeft className="w-3.5 h-3.5" />
                    <span>Previous</span>
                  </button>

                  <div className="px-2.5 py-1 text-muted-foreground text-[11px] font-mono">
                    Page <span className="font-semibold text-foreground">{page}</span> of{" "}
                    <span className="font-semibold text-foreground">{totalPages}</span>
                  </div>

                  <button
                    type="button"
                    disabled={page >= totalPages || isLoadingUsers}
                    onClick={() => setPage((prev) => prev + 1)}
                    className="px-2.5 py-1.5 rounded-lg border border-border bg-surface hover:bg-surface-container text-foreground disabled:opacity-40 disabled:pointer-events-none transition-colors flex items-center gap-1 cursor-pointer font-medium"
                  >
                    <span>Next</span>
                    <ChevronRight className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* TAB 2: PENDING UPGRADE REQUESTS */}
      {activeTab === "requests" && (
        <div className="space-y-4">
          <div className="p-4 rounded-xl bg-surface border border-border flex items-center justify-between">
            <div>
              <h3 className="text-sm font-bold text-foreground">User Upgrade Requests</h3>
              <p className="text-xs text-muted-foreground mt-0.5">
                Users can request access from their application settings. Approving instantly elevates them to the Pro tier.
              </p>
            </div>
            <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-surface-container border border-border">
              {pendingCount} Pending
            </span>
          </div>

          {isLoadingRequests ? (
            <div className="py-12 text-center text-muted-foreground">
              <RefreshCw className="w-4 h-4 animate-spin mx-auto mb-2 text-primary" />
              Loading pending requests...
            </div>
          ) : requests.length === 0 ? (
            <div className="py-16 text-center border border-dashed border-border rounded-xl bg-surface/50 space-y-2">
              <CheckCircle2 className="w-8 h-8 text-emerald-400 mx-auto opacity-70" />
              <h4 className="text-sm font-semibold text-foreground">No Pending Upgrade Requests</h4>
              <p className="text-xs text-muted-foreground max-w-sm mx-auto">
                All team and user requests have been processed. New requests will appear here in real time.
              </p>
            </div>
          ) : (
            <div className="grid gap-3">
              {requests.map((req) => (
                <div
                  key={req.id}
                  className="p-5 rounded-xl border border-border bg-surface hover:border-primary/30 transition-all flex flex-col md:flex-row md:items-center justify-between gap-4 shadow-sm"
                >
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-bold text-foreground">{req.user_name || "Unknown"}</span>
                      <span className="text-xs text-muted-foreground font-mono">({req.user_email})</span>
                      <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/20">
                        Wants {req.requested_role}
                      </span>
                    </div>

                    <p className="text-xs text-muted-foreground italic bg-surface-container/40 p-2 rounded-lg border border-border/40 mt-1">
                      &ldquo;{req.reason || "User requested upgrade via settings"}&rdquo;
                    </p>

                    <div className="flex items-center gap-3 text-[11px] text-muted-foreground pt-1">
                      <span className="flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        Submitted {new Date(req.created_at).toLocaleString()}
                      </span>
                      <span>&bull;</span>
                      <span>Current: <strong className="text-foreground capitalize">{req.current_role}</strong></span>
                    </div>
                  </div>

                  <div className="flex items-center gap-2 shrink-0">
                    <button
                      type="button"
                      disabled={approveMutation.isPending}
                      onClick={() => approveMutation.mutate(req.id)}
                      className="px-4 py-2 rounded-lg text-xs font-semibold bg-emerald-600 hover:bg-emerald-500 text-white transition-colors cursor-pointer flex items-center gap-1.5 shadow-sm active:scale-95"
                    >
                      <Check className="w-3.5 h-3.5" />
                      Approve &amp; Make Pro
                    </button>
                    <button
                      type="button"
                      disabled={rejectMutation.isPending}
                      onClick={() => rejectMutation.mutate(req.id)}
                      className="px-3 py-2 rounded-lg text-xs font-medium text-muted-foreground hover:bg-danger-subtle hover:text-danger transition-colors cursor-pointer flex items-center gap-1"
                    >
                      <X className="w-3.5 h-3.5" />
                      Reject
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
