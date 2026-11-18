import dashboardData from "@/data/dashboard-data.json";
import Dashboard from "@/components/dashboard/dashboard";
import type { DashboardData } from "@/components/dashboard/types";

export default function Home() {
  return <Dashboard data={dashboardData as unknown as DashboardData} />;
}
