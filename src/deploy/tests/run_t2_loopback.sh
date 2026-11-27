#!/usr/bin/env bash
# T2: the whole wireless stack on loopback, no robot and no WiFi:
#   src/deploy/fake_robot.py (synthetic robot, DDS on lo)  <->  src/deploy/relay_jetson.py  <->  src/deploy/deploy_real_wireless.py
# Injects the failures that matter and asserts on what the fake robot actually receives. Run inside the project
# container (it has the Unitree SDK and torch), isolated from every network:
#   docker run --rm --network none --entrypoint bash -v "$PWD:/repo:ro" quadruped_go2_locomotion:latest \
#       -lc 'bash /repo/deploy/tests/run_t2_loopback.sh'
set -u
# Logs go to files: without this Python block-buffers them, and the assertions below would read stale tails.
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/../../.."
PY="${PY:-/workspace/isaaclab/_isaac_sim/python.sh}"
KEY=/tmp/t2_relay.key
rm -f "$KEY"
$PY src/deploy/relay_jetson.py --init_key "$KEY" >/dev/null 2>&1 || { echo "could not create a key"; exit 2; }
PASS=0; FAIL=0

ok() { echo "  PASS: $1"; PASS=$((PASS + 1)); }
bad() { echo "  FAIL: $1"; FAIL=$((FAIL + 1)); }
has() { if grep -qE "$3" "$2"; then ok "$1"; else bad "$1   [no /$3/ in $2]"; fi; }
hasnt() { if grep -qE "$3" "$2"; then bad "$1   [unexpected /$3/ in $2]"; else ok "$1"; fi; }
stopall() {
    pkill -CONT -f deploy_real_wireless.py 2>/dev/null
    pkill -9 -f "src/deploy/fake_robot.py" 2>/dev/null; pkill -9 -f relay_jetson.py 2>/dev/null
    pkill -9 -f deploy_real_wireless.py 2>/dev/null; sleep 2
}
start_fake() { $PY src/deploy/fake_robot.py --network_interface lo --responsive --rate_hz 200 "$@" >/tmp/fake.log 2>&1 & sleep 7; }
start_relay() { $PY src/deploy/relay_jetson.py --network_interface lo --key_file "$KEY" "$@" >/tmp/relay.log 2>&1 & sleep 7; }
start_client() {
    $PY src/deploy/deploy_real_wireless.py --relay_host 127.0.0.1 --relay_key_file "$KEY" --skip_release \
        --no_pause_for_hoist --startup_ramp_s 1 --network_interface lo >/tmp/client.log 2>&1 &
}
lowcmd_count() { grep -c "received LowCmd" /tmp/fake.log; }
last_kp0() { grep "received LowCmd" /tmp/fake.log | tail -1 | sed -E 's/.*kp0=([0-9.]+).*/\1/'; }
reset() { stopall; : >/tmp/fake.log; : >/tmp/relay.log; : >/tmp/client.log; }

echo "=== A. happy path: connect, authenticate, ramp, run, clean stop ==="
reset; start_fake; start_relay; start_client
sleep 16
has "client authenticated the relay (mutual)" /tmp/client.log "Relay authenticated \(mutual\)"
has "client got robot state through the relay" /tmp/client.log "Robot state stream connected \(via the relay\)"
has "client entered the control loop" /tmp/client.log "Entering control loop"
has "relay went ACTIVE and is receiving commands" /tmp/relay.log "status: ACTIVE +cmds=[1-9][0-9]{2,}"
has "fake robot receives LowCmd with stiffness (kp > 0)" /tmp/fake.log "kp0=[1-9][0-9]*\.[0-9]+"
has "fake robot sees the LowCmd header bytes (254,239); the CRC value is logged, not verified, by the fake" /tmp/fake.log "received LowCmd #[0-9]+: head=\[254, 239\]"
pkill -TERM -f deploy_real_wireless.py; sleep 5
has "client reports the command->state round trip" /tmp/client.log "round trip over [0-9]+ samples"
hasnt "client had no fault or e-stop" /tmp/client.log "\[FAULT\]|\[ESTOP\]|Traceback"
has "relay damped when the desktop shut down" /tmp/relay.log "DAMPING .*desktop"
sleep 2; c1=$(lowcmd_count); sleep 2; c2=$(lowcmd_count)
[ "$c1" = "$c2" ] && ok "relay went quiet after damping (LowCmd count stopped growing at $c2)" || bad "relay still publishing ($c1 -> $c2)"
hasnt "relay had no traceback" /tmp/relay.log "Traceback"

echo "=== B. WiFi stall: sender frozen, TCP stays open (relay must damp on its own) ==="
reset; start_fake; start_relay; start_client; sleep 16
pkill -STOP -f deploy_real_wireless.py; sleep 3
has "relay damped on command silence" /tmp/relay.log "DAMPING .*no valid command for"
[ "$(last_kp0)" = "0.00" ] && ok "last LowCmd the robot saw has kp=0 (damped)" || bad "last LowCmd kp0=$(last_kp0), expected 0.00"
c1=$(lowcmd_count); sleep 2; c2=$(lowcmd_count)
[ "$c1" = "$c2" ] && ok "relay stopped publishing after the damp hold" || bad "relay still publishing ($c1 -> $c2)"
pkill -CONT -f deploy_real_wireless.py

echo "=== C. client killed outright (kill -9): relay must damp from the dropped connection ==="
reset; start_fake; start_relay; start_client; sleep 16
pkill -9 -f deploy_real_wireless.py; sleep 3
has "relay damped on the lost control connection" /tmp/relay.log "DAMPING .*control connection"
[ "$(last_kp0)" = "0.00" ] && ok "last LowCmd has kp=0" || bad "last LowCmd kp0=$(last_kp0)"

echo "=== D. robot state stream dies: relay must damp ==="
reset; start_fake; start_relay; start_client; sleep 16
pkill -9 -f "src/deploy/fake_robot.py"; sleep 3
has "relay damped when the robot's LowState stopped" /tmp/relay.log "DAMPING .*no LowState from the robot"
has "client stopped on the link-lost guard" /tmp/client.log "\[LINK\]"

echo "=== E. controller e-stop from the robot reaches the desktop and damps ==="
reset; start_fake --press_estop_after 26; start_relay; start_client; sleep 30
has "client saw the e-stop" /tmp/client.log "\[ESTOP\]"
has "relay damped" /tmp/relay.log "DAMPING"

echo "=== F. security: wrong key, forged packets, invalid authenticated command ==="
reset; start_fake; start_relay
$PY - <<'EOF' >/tmp/sec.log 2>&1
import os, sys, socket, time
sys.path.insert(0, "src/deploy")
import relay_link as rl
good = rl.load_key("/tmp/t2_relay.key")
bad = bytes(range(32))

class S:  # minimal state sink
    ready = False
    def update_from_low_state(self, m): pass
    def update_from_sportmode_state(self, m): pass

c = rl.RelayClient("127.0.0.1", bad, S(), list(range(12)))
try:
    c.connect(3.0); print("WRONGKEY: ACCEPTED")
except rl.RelayError as e:
    print("WRONGKEY: refused")
except Exception as e:
    print("WRONGKEY: refused (", type(e).__name__, ")")

# forged UDP command with a bad tag straight at the command port: must be ignored
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.sendto(rl.pack_cmd(bad, 1, 1, 1, [0.0] * 12, 25.0, 0.5), ("127.0.0.1", rl.DEFAULT_CMD_PORT))
s.sendto(os.urandom(120), ("127.0.0.1", rl.DEFAULT_CMD_PORT))
print("FORGED: sent")

# a legitimate client, then an AUTHENTICATED but invalid command (kp far above the limit): relay must damp
c = rl.RelayClient("127.0.0.1", good, S(), list(range(12)))
c.connect(3.0); print("GOODKEY: connected")
c.request_release(False, True, lambda: None, timeout_s=10.0); print("RELEASE: ok")
c._udp_cmd.sendto(rl.pack_cmd(good, c.session, 1, time.monotonic_ns(), [0.0] * 12, 100.0, 0.5), ("127.0.0.1", c.cmd_port))
time.sleep(1.0)
print("INVALID: sent")
c.close()
EOF
has "wrong key is refused by the client" /tmp/sec.log "WRONGKEY: refused"
has "relay logged the rejected login" /tmp/relay.log "Rejected a connection"
has "a legitimate client still connects afterwards" /tmp/sec.log "GOODKEY: connected"
has "release (skip, loopback) succeeded" /tmp/sec.log "RELEASE: ok"
has "forged and random UDP packets were counted as rejected" /tmp/relay.log "rejected\(auth/stale/invalid\)=[1-9]|status: ACTIVE.*rejected\(auth/stale/invalid\)=[1-9]|Rejected"
has "authenticated-but-invalid command (kp=100) damps the robot" /tmp/relay.log "DAMPING .*failed validation"

stopall
echo
echo "RESULT: $PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
