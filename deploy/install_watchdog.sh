#!/usr/bin/env bash
# Ship and arm eval/watchdog.py on the orchestrator box.
#
# WHY A SCRIPT AND NOT A README. The node's /workspace/ralph-v2 is rsync-deployed and is NOT a
# checkout, so "it is committed" proves nothing about what is running. Two of the seven round-1
# failures were exactly that gap — a fix that existed in the repo and not on the box — so every
# step here either copies bytes or proves the bytes arrived.
#
# READ-ONLY UNTIL STEP 6. Steps 1-5 ship and verify; 6 is the live acceptance read; 7 arms it.
set -eu

N=${RALPH_NODE:-root@64.34.93.55}
K=${RALPH_SSH_KEY:-/root/.ssh/id_bitzic}
REPO=${RALPH_REPO:-/workspace/unicorn/karpathian/ralph-v2}
DEST=${RALPH_DEST:-/workspace/ralph-v2}
S() { ssh -i "$K" -o StrictHostKeyChecking=no -o ConnectTimeout=15 "$N" "$@"; }

echo "== 1. the suite, here"
( cd "$REPO" && python3 -m tests.test_watchdog && python3 -m tests.test_progress_timeout \
    && python3 -m tests.test_crown_path >/dev/null && echo "  crown path OK" )

echo "== 2. ship"
rsync -az --exclude '.git/' --exclude '.venv/' --exclude '.env' --exclude 'runs/' \
      --exclude '__pycache__/' -e "ssh -i $K -o StrictHostKeyChecking=no" \
      "$REPO/eval" "$REPO/tests" "$REPO/deploy" "$N:$DEST/"

echo "== 3. prove the bytes arrived"
( cd "$REPO" && md5sum eval/watchdog.py eval/orchestrator.py eval/run_orchestrated.py )
S "cd $DEST && md5sum eval/watchdog.py eval/orchestrator.py eval/run_orchestrated.py"

echo "== 4. the suite, on the box it will guard, with the box's own venv"
S "cd $DEST && .venv/bin/python -m tests.test_watchdog | tail -1"
S "cd $DEST && .venv/bin/python -m tests.test_progress_timeout | tail -1"

echo "== 5. the validator drop-in (never a unit rewrite: the node's unit is not in git)"
S "mkdir -p /etc/systemd/system/ralph-validator.service.d"
scp -i "$K" -o StrictHostKeyChecking=no \
    "$REPO/deploy/ralph-validator.service.d/10-stop-timeout.conf" \
    "$N:/etc/systemd/system/ralph-validator.service.d/"
scp -i "$K" -o StrictHostKeyChecking=no \
    "$REPO"/deploy/ralph-watchdog.service "$REPO"/deploy/ralph-watchdog.timer \
    "$N:/etc/systemd/system/"
S "systemctl daemon-reload && systemctl show ralph-validator.service -p TimeoutStopUSec"
#   ^ must read 15min. Asserted against what RUNS, not against a file in the repo.

echo "== 6. live acceptance — the seam no test can reach"
S "cd $DEST && .venv/bin/python -m eval.watchdog status"
S "cd $DEST && .venv/bin/python -m eval.watchdog once --dry-run"

echo
echo "== 7. NOT DONE AUTOMATICALLY — arming is a decision, in this order:"
echo "   systemctl enable --now ralph-watchdog.timer      # ORPHAN/OVERDUE armed, HUNG observing"
echo "   systemctl start --no-block ralph-validator       # round 1 attempt 8, THROUGH THE UNIT"
echo "   journalctl -u ralph-watchdog -f                  # confirm RUNNING through rented+install"
echo "   systemctl edit ralph-watchdog.service            # then RALPH_WATCHDOG_HUNG_ACT=1"
echo
echo "   Never 'python -m eval.run_orchestrated' in a shell: that round writes to an arbitrary"
echo "   path, is invisible to the HUNG rung, and is bounded only by the 2.75 h rental ceiling."
