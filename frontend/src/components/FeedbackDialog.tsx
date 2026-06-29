import React, { useState, useEffect } from "react";
import { X, Star } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "../lib/utils";

interface FeedbackDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (rating: number, comment: string) => void;
  defaultRating: number;
}

export default function FeedbackDialog({
  isOpen,
  onClose,
  onSubmit,
  defaultRating,
}: FeedbackDialogProps) {
  const [rating, setRating] = useState(defaultRating);
  const [comment, setComment] = useState("");
  const [hoverRating, setHoverRating] = useState<number | null>(null);

  // Sync default rating when modal opens
  useEffect(() => {
    if (isOpen) {
      setRating(defaultRating);
      setComment("");
    }
  }, [isOpen, defaultRating]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(rating, comment);
    onClose();
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 select-none">
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="absolute inset-0 bg-background/60 backdrop-blur-sm"
          />

          {/* Modal Content */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 15 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 15 }}
            transition={{ type: "spring", duration: 0.4 }}
            className="relative w-full max-w-md bg-surface border border-border rounded-2xl p-6 shadow-2xl z-10 flex flex-col space-y-5"
          >
            {/* Header */}
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-bold text-foreground tracking-tight">Submit Answer Feedback</h3>
              <button
                onClick={onClose}
                className="p-1 text-muted-foreground hover:text-foreground hover:bg-surface-container rounded-lg transition-colors cursor-pointer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <form onSubmit={handleSubmit} className="space-y-4">
              {/* Rating representation */}
              <div className="flex flex-col items-center space-y-2">
                <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">How was this response?</span>
                <div className="flex items-center space-x-1.5">
                  {[1, 2, 3, 4, 5].map((star) => (
                    <button
                      key={star}
                      type="button"
                      onClick={() => setRating(star)}
                      onMouseEnter={() => setHoverRating(star)}
                      onMouseLeave={() => setHoverRating(null)}
                      className="p-1 rounded-md transition-transform active:scale-90 hover:bg-surface-container"
                    >
                      <Star
                        className={cn(
                          "w-7 h-7 stroke-1 transition-colors cursor-pointer",
                          star <= (hoverRating ?? rating)
                            ? "fill-amber-400 text-amber-400"
                            : "text-muted-foreground hover:text-amber-300"
                        )}
                      />
                    </button>
                  ))}
                </div>
                <span className="text-xs font-medium text-muted-foreground h-4">
                  {rating === 1 && "Poor / Incorrect"}
                  {rating === 2 && "Fair / Needs Work"}
                  {rating === 3 && "Average"}
                  {rating === 4 && "Good / High Quality"}
                  {rating === 5 && "Excellent / Perfect"}
                </span>
              </div>

              {/* Comment Text Area */}
              <div className="flex flex-col space-y-1.5">
                <label className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Additional details (Optional)</label>
                <textarea
                  value={comment}
                  onChange={(e) => setComment(e.target.value)}
                  placeholder="Tell us what was good or what could be improved..."
                  rows={4}
                  className="w-full text-sm bg-background border border-border rounded-xl px-4 py-3 text-foreground placeholder:text-muted-foreground/60 focus:outline-none focus:ring-1 focus:ring-primary/20 resize-none font-medium leading-relaxed"
                />
              </div>

              {/* Actions */}
              <div className="flex items-center justify-end space-x-3 pt-2">
                <button
                  type="button"
                  onClick={onClose}
                  className="px-4 py-2 text-sm font-semibold text-muted-foreground hover:text-foreground hover:bg-surface-container rounded-xl transition-all cursor-pointer"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-5 py-2 text-sm font-semibold bg-primary hover:bg-primary/90 text-white rounded-xl transition-all shadow-md shadow-primary/10 active:scale-95 cursor-pointer"
                >
                  Submit Feedback
                </button>
              </div>
            </form>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
}
