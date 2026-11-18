#!/bin/bash

INPUT_SHAPE="$1"
CONFIG="$2"

SESSION="foldaball"

# Destroy old session so the layout is rebuilt every time
tmux kill-session -t "$SESSION" 2>/dev/null

# Create session
tmux new-session -d -s "$SESSION"

# Create three additional panes
tmux split-window -h -t "$SESSION"
tmux split-window -v -t "$SESSION"
tmux split-window -v -t "$SESSION"

# Force an evenly tiled layout
tmux select-layout -t "$SESSION" tiled

# Get actual pane IDs
mapfile -t PANES < <(tmux list-panes -t "$SESSION" -F '#{pane_id}')

# Initialize each pane
for pane in "${PANES[@]}"; do
    tmux send-keys -t "$pane" "cd ../../" C-m
    tmux send-keys -t "$pane" "tiny -r optimize \"../../assets/test_shapes/$INPUT_SHAPE\" -c \"../../$CONFIG\"" C-m
done

# Attach
tmux attach-session -t "$SESSION"


