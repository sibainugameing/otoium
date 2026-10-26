/// A linear undo/redo stack.
///
/// Stores copies of T (cloned snapshots). When a new item is pushed
/// after an undo, the redo branch is discarded (standard linear history).
pub struct History<T: Clone> {
    max_depth: usize,
    /// Past states (oldest at index 0, most recent at back).
    past: Vec<T>,
    /// Future states after an undo (most recently undone at back).
    future: Vec<T>,
}

impl<T: Clone> History<T> {
    pub fn new(max_depth: usize) -> Self {
        Self {
            max_depth,
            past: Vec::new(),
            future: Vec::new(),
        }
    }

    /// Push a snapshot onto the history. Clears any pending redo states.
    pub fn push(&mut self, state: T) {
        self.future.clear();
        self.past.push(state);
        if self.past.len() > self.max_depth {
            self.past.remove(0);
        }
    }

    /// Undo: pop the most recent past state and return it.
    /// The caller should push the current state to `future` before applying.
    pub fn undo(&mut self) -> Option<T> {
        self.past.pop().map(|state| {
            state
        })
    }

    /// Redo: pop the most recently undone state and return it.
    pub fn redo(&mut self) -> Option<T> {
        self.future.pop()
    }

    /// Store current state in future before applying an undo.
    /// (Call this before applying the undo result to the current state.)
    pub fn save_for_redo(&mut self, state: T) {
        self.future.push(state);
    }

    pub fn can_undo(&self) -> bool {
        !self.past.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.future.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_undo_redo() {
        let mut h: History<i32> = History::new(10);
        h.push(1);
        h.push(2);
        h.push(3);

        assert_eq!(h.undo(), Some(3));
        h.save_for_redo(3);
        assert_eq!(h.undo(), Some(2));
        h.save_for_redo(2);
        assert_eq!(h.redo(), Some(2));

        // New push clears redo
        h.push(10);
        assert!(!h.can_redo());
    }

    #[test]
    fn test_max_depth() {
        let mut h: History<i32> = History::new(3);
        for i in 0..10 {
            h.push(i);
        }
        // Should only keep last 3
        assert_eq!(h.undo(), Some(9));
        assert_eq!(h.undo(), Some(8));
        assert_eq!(h.undo(), Some(7));
        assert_eq!(h.undo(), None);
    }
}
