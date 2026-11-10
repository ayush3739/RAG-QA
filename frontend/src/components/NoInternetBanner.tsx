import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { WifiOff, Wifi } from "lucide-react";

export default function NoInternetBanner() {
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [showReconnected, setShowReconnected] = useState(false);

  useEffect(() => {
    const handleOnline = () => {
      setIsOnline(true);
      setShowReconnected(true);
      setTimeout(() => setShowReconnected(false), 3000);
    };
    const handleOffline = () => {
      setIsOnline(false);
      setShowReconnected(false);
    };

    window.addEventListener("online", handleOnline);
    window.addEventListener("offline", handleOffline);
    return () => {
      window.removeEventListener("online", handleOnline);
      window.removeEventListener("offline", handleOffline);
    };
  }, []);

  return (
    <AnimatePresence>
      {(!isOnline || showReconnected) && (
        <motion.div
          key={isOnline ? "reconnected" : "offline"}
          initial={{ y: -60, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          exit={{ y: -60, opacity: 0 }}
          transition={{ type: "spring", stiffness: 350, damping: 30 }}
          className="fixed top-0 left-0 right-0 z-[9999] flex items-center justify-center p-3 pointer-events-none"
        >
          <div
            className={`flex items-center gap-2.5 px-4 py-2.5 rounded-xl border text-xs font-semibold shadow-sm bg-surface-container-lowest
              ${isOnline
                ? "border-success-subtle text-success"
                : "border-danger-subtle text-danger"
              }`}
          >
            {isOnline ? (
              <>
                <Wifi className="w-4 h-4 text-success shrink-0" />
                <span>Connection restored</span>
              </>
            ) : (
              <>
                <WifiOff className="w-4 h-4 text-danger shrink-0" />
                <span>Network offline: requests will fail until connection is restored.</span>
              </>
            )}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
