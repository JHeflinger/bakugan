
#!/bin/bash

# Validate arguments
if [ "$#" -ne 1 ]; then
    echo "Usage: ./multirun <object>"
    exit 1
fi

OBJECT="$(realpath -- "$1")"

# Find directory containing this script
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

CONFIG_DIR="$SCRIPT_DIR/multirun_configs"
SESSION="foldaball"

# Find all config files
shopt -s nullglob
CONFIGS=("$CONFIG_DIR"/*)

# Remove directories from config list
FILES=()
for config in "${CONFIGS[@]}"; do
    if [ -f "$config" ]; then
        FILES+=("$config")
    fi
done

# Check for configs
if [ "${#FILES[@]}" -eq 0 ]; then
    echo "No config files found in $CONFIG_DIR"
    exit 1
fi

echo "Found ${#FILES[@]} configurations"

# Check if session is already running
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "A tmux session named '$SESSION' is already live."
    exit 1
fi

# Create session
tmux new-session -d -s "$SESSION"

# Create one pane per remaining config
for ((i=1; i<${#FILES[@]}; i++)); do
    tmux split-window -d -t "$SESSION"
done

# Arrange panes
tmux select-layout -t "$SESSION" tiled

# Get pane IDs
mapfile -t PANES < <(tmux list-panes -t "$SESSION" -F '#{pane_id}')

# Launch optimization in each pane
for i in "${!PANES[@]}"; do
    pane="${PANES[$i]}"
    config="${FILES[$i]}"

    echo "Launching $OBJECT with $config"

    tmux send-keys -t "$pane" "cd ../../" C-m
    tmux send-keys -t "$pane" "tiny -r optimize \"../../assets/test_shapes/$INPUT_SHAPE\" -c \"../../multirun_configs/$config\"" C-m
done

# Attach
tmux attach-session -t "$SESSION"

