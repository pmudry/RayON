#!/usr/bin/env bash
# Launch a headless VNC desktop and RayON (interactive CUDA mode) inside it.
#
# Usage:  scripts/rayon_vnc.sh [rayon options...]
#   Default resolution is 1080p (-r 1080). Any option is passed to rayon,
#   e.g.  scripts/rayon_vnc.sh -r 720 --scene scenes/foo.yaml
#
# The VNC screen is sized to the render resolution and the window is placed
# undecorated at (0,0), so a viewer in full screen shows exactly the render.
# Ctrl-C (or rayon exiting) stops everything: rayon, openbox and the VNC server.
#
# Needs: tigervnc-standalone-server, openbox  (sudo apt install ...)

set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_DIR/build"
RAYON="$BUILD_DIR/rayon"

[[ -x "$RAYON" ]] || { echo "rayon not found at $RAYON (build it first)" >&2; exit 1; }
command -v Xtigervnc >/dev/null || { echo "Xtigervnc missing: sudo apt install tigervnc-standalone-server" >&2; exit 1; }
command -v openbox   >/dev/null || { echo "openbox missing: sudo apt install openbox" >&2; exit 1; }

# ---- resolution: find "-r <value>" in the args, default 1080 -------------
ARGS=("$@")
RES=""
for ((i = 0; i < ${#ARGS[@]}; i++)); do
   [[ "${ARGS[$i]}" == "-r" && $((i + 1)) -lt ${#ARGS[@]} ]] && RES="${ARGS[$((i + 1))]}"
done
if [[ -z "$RES" ]]; then
   RES=1080
   ARGS=(-r "$RES" "${ARGS[@]}")
fi
if [[ "$RES" == *x* ]]; then
   WIDTH="${RES%x*}"; HEIGHT="${RES#*x}"
else
   HEIGHT="$RES"; WIDTH=$((HEIGHT * 16 / 9))
fi
GEOMETRY="${WIDTH}x${HEIGHT}"

# ---- pick a free X display ----------------------------------------------
DISP=1
while [[ -e "/tmp/.X11-unix/X$DISP" || -e "/tmp/.X$DISP-lock" ]]; do DISP=$((DISP + 1)); done
VNC_PORT=$((5900 + DISP))

# ---- cleanup: kill everything we started --------------------------------
RAYON_PID=""; WM_PID=""; VNC_PID=""
cleanup() {
   trap - EXIT INT TERM
   echo
   echo "Stopping rayon, openbox and the VNC server..."
   for pid in "$RAYON_PID" "$WM_PID" "$VNC_PID"; do
      [[ -n "$pid" ]] && kill "$pid" 2>/dev/null
   done
   sleep 1
   for pid in "$RAYON_PID" "$WM_PID" "$VNC_PID"; do
      [[ -n "$pid" ]] && kill -9 "$pid" 2>/dev/null
   done
   rm -rf "$TMP"
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT TERM

TMP="$(mktemp -d)"

# openbox config: no decorations, window at (0,0), nothing else
cat > "$TMP/rc.xml" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<openbox_config xmlns="http://openbox.org/3.4/rc">
  <theme><keepBorder>no</keepBorder></theme>
  <applications>
    <application class="*">
      <decor>no</decor>
      <position force="yes"><x>0</x><y>0</y></position>
    </application>
  </applications>
</openbox_config>
EOF

# ---- start VNC server + window manager ----------------------------------
Xtigervnc ":$DISP" -geometry "$GEOMETRY" -depth 24 -localhost yes \
   -SecurityTypes None -desktop "RayON on $(hostname)" -rfbport "$VNC_PORT" \
   > "$TMP/xvnc.log" 2>&1 &
VNC_PID=$!
for _ in $(seq 1 50); do [[ -e "/tmp/.X11-unix/X$DISP" ]] && break; sleep 0.1; done
[[ -e "/tmp/.X11-unix/X$DISP" ]] || { echo "VNC server failed to start:"; cat "$TMP/xvnc.log"; exit 1; }

export DISPLAY=":$DISP"
openbox --config-file "$TMP/rc.xml" > "$TMP/openbox.log" 2>&1 &
WM_PID=$!
sleep 0.5

# ---- how to connect -----------------------------------------------------
HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
cat <<EOF
==============================================================
 VNC display :$DISP  (${GEOMETRY}, localhost:$VNC_PORT, no password)

 Connect from your machine, with TigerVNC viewer:
   vncviewer -via $USER@$HOST_IP localhost:$DISP

 Or through a manual ssh tunnel:
   ssh -N -f -L $VNC_PORT:localhost:$VNC_PORT $USER@$HOST_IP
   vncviewer localhost:$VNC_PORT

 Ctrl-C here stops rayon and the VNC server.
==============================================================
EOF

# ---- run rayon (foreground via wait, so Ctrl-C reaches the trap) --------
cd "$BUILD_DIR"
"$RAYON" "${ARGS[@]}" &
RAYON_PID=$!
wait "$RAYON_PID"
STATUS=$?
RAYON_PID=""
echo "rayon exited with status $STATUS"
exit "$STATUS"
