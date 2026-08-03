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
          initial={{ y: -80, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          exit={{ y: -80, opacity: 0 }}
          transition={{ type: "spring", stiffness: 300, damping: 28 }}
          className="fixed top-0 left-0 right-0 z-[9999] flex items-center justify-center"
        >
          <div
            className={`mt-3 flex items-center gap-3 px-5 py-3 rounded-xl shadow-2xl border text-sm font-medium backdrop-blur-md
              ${isOnline
                ? "bg-emerald-950/90 border-emerald-500/30 text-emerald-300"
                : "bg-zinc-950/90 border-red-500/30 text-red-300"
              }`}
          >
            {isOnline ? (
              <>
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ type: "spring", stiffness: 400, damping: 15 }}
                >
                  <Wifi className="w-4 h-4 text-emerald-400" />
                </motion.div>
                <span>You're back online</span>
              </>
            ) : (
              <>
                <motion.div
                  animate={{ opacity: [1, 0.4, 1] }}
                  transition={{ duration: 1.5, repeat: Infinity }}
                >
                  <WifiOff className="w-4 h-4 text-red-400" />
                </motion.div>
                <span>No internet connection — requests will fail until reconnected</span>
              </>
            )}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
