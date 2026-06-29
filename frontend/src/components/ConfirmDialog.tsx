import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { AlertTriangle } from "lucide-react";

interface ConfirmDialogProps {
  isOpen: boolean;
  title: string;
  description: string;
  confirmLabel?: string;
  cancelLabel?: string;
  confirmClassName?: string;
  onConfirm: () => void;
  onCancel: () => void;
}

export default function ConfirmDialog({
  isOpen,
  title,
  description,
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  confirmClassName = "bg-red-500 hover:bg-red-600 text-white",
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-background/60 backdrop-blur-sm flex items-center justify-center p-4 z-[100]"
          onClick={onCancel}
        >
          <motion.div
            initial={{ scale: 0.93, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.93, opacity: 0 }}
            transition={{ type: "spring", stiffness: 400, damping: 30 }}
            className="bg-surface-container-lowest max-w-md w-full rounded-2xl border border-border shadow-premium p-6 space-y-5"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-start space-x-4">
              <div className="p-2.5 bg-red-500/10 border border-red-500/20 rounded-xl shrink-0">
                <AlertTriangle className="w-5 h-5 text-red-500" />
              </div>
              <div>
                <h3 className="text-base font-bold text-foreground tracking-tight">{title}</h3>
                <p className="text-sm text-muted-foreground mt-1.5 leading-relaxed">{description}</p>
              </div>
            </div>

            <div className="flex items-center space-x-3 pt-1">
              <button
                onClick={onCancel}
                className="flex-1 py-2.5 px-4 rounded-xl border border-border bg-surface hover:bg-surface-container text-foreground font-semibold text-sm transition-all cursor-pointer"
              >
                {cancelLabel}
              </button>
              <button
                onClick={onConfirm}
                className={`flex-1 py-2.5 px-4 rounded-xl font-semibold text-sm transition-all cursor-pointer ${confirmClassName}`}
              >
                {confirmLabel}
              </button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
