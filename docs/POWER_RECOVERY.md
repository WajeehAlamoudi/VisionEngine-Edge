<div align="center">

```
██╗   ██╗██╗███████╗██╗ ██████╗ ███╗   ██╗    ███████╗██████╗  ██████╗ ███████╗
██║   ██║██║██╔════╝██║██╔═══██╗████╗  ██║    ██╔════╝██╔══██╗██╔════╝ ██╔════╝
██║   ██║██║███████╗██║██║   ██║██╔██╗ ██║    █████╗  ██║  ██║██║  ███╗█████╗
╚██╗ ██╔╝██║╚════██║██║██║   ██║██║╚██╗██║    ██╔══╝  ██║  ██║██║   ██║██╔══╝
 ╚████╔╝ ██║███████║██║╚██████╔╝██║ ╚████║    ███████╗██████╔╝╚██████╔╝███████╗
  ╚═══╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝   ╚══════╝╚═════╝  ╚═════╝ ╚══════╝
```

### **Power Recovery**

<br/>

[![Platform](https://img.shields.io/badge/Platform-Jetson%20P3768-1a1a2e?style=for-the-badge&logoColor=4fc3f7)](#scope-and-recommended-behavior)
[![Hardware](https://img.shields.io/badge/Hardware-Yahboom%20case-1a1a2e?style=for-the-badge&logoColor=4fc3f7)](#yahboom-case-jumper)
[![Recovery](https://img.shields.io/badge/Recovery-Auto%20power--on-1a1a2e?style=for-the-badge&logoColor=4fc3f7)](#test-recovery)

<br/>

> *Why an edge device stays off after an outage, which jumper controls startup,
> and how to verify recovery when power returns.*

</div>

---

## When to use this guide

Use this guide when the entire edge device goes offline and needs a physical
power-button press to return. A reachable NVR does not prove that the edge
device still has power: they may use different adapters or power paths.

---

## Scope and recommended behavior

The wiring below applies to the NVIDIA **P3768 reference carrier board** used
with Jetson Orin Nano/NX and a compatible Yahboom aluminum case. Other Jetson
carriers and non-Jetson edges require their manufacturer's instructions.
The Linux device-tree model identifies the installed software configuration;
confirm the physical board and labels before using this pin mapping.

For an unattended shop deployment, retain normal button behavior and enable
automatic startup when DC power returns. Do not disable the desktop or remap
the button merely to recover from a power outage.

| Event | Expected behavior |
|---|---|
| Complete DC power loss, then power restored | Board boots automatically; enabled services start |
| Manual shutdown while DC remains connected | Board stays off until the power button is pressed or DC is cycled |
| `sudo reboot` | Board restarts without cycling DC |
| Long power-button hold | May force a hardware power-off; software cannot guarantee recovery |

Auto-power-on responds to restored DC power. It does not continuously force
the board to remain running. A Linux service cannot execute after the board
has powered off. A UPS addresses interruptions; the jumper addresses startup
after an interruption.

---

## Yahboom case jumper

Yahboom's case installation guide instructs users to bridge `DISABLE` and
`AUTO ON` so the case button controls startup. On the P3768 J14 header, that
jumper **disables** automatic startup.

| J14 pins | Function | For automatic startup |
|---|---|---|
| 1–2 | Case LED: negative on 1, positive on 2 | Keep connected |
| 5–6 | Auto-power-on disable jumper | Leave open |
| 7–8 | Momentary reset button | Leave existing correct wiring alone |
| 9–10 | Force recovery | Do not bridge for normal operation |
| 11–12 | Momentary power button | Keep connected |

Jumper color is not an identifier. Verify the `DISABLE / AUTO ON` labels and
pin orientation; do not remove a UART connector or guess from wire color.

1. Shut down from the Jetson terminal with `sudo shutdown -h now`.
2. Wait for shutdown to finish, then disconnect DC power before touching wiring.
3. Remove only the jumper bridging J14 pins 5–6 (`DISABLE / AUTO ON`).
4. Keep the correctly connected case button and LED cables attached.
5. Reconnect DC power without pressing the case button.

---

## Test recovery

Perform this test on site. Shut down cleanly first instead of deliberately
cutting power during database writes.

1. Run `sudo shutdown -h now` and wait for shutdown to finish.
2. Unplug the DC connector **at the Jetson**, wait 30 seconds, then reconnect it.
   Disconnecting at the board avoids residual output from the power adapter.
3. Do not press the button. Allow 60–90 seconds for boot and networking.
4. From a PC on the same tailnet, run these PowerShell commands. Substitute
   your device hostname and Linux username if different:

```powershell
tailscale ping jetson-yahboom
tailscale ssh jetson@jetson-yahboom "systemctl is-active tailscaled visionengine-edge"
```

Expect a `pong` and two `active` lines. Initial ping timeouts can occur during
boot. An illuminated LED alone does not prove Linux or the application started.

On the Jetson, verify that services are enabled for future boots and that all
configured cameras have resumed processing:

```bash
systemctl is-enabled tailscaled visionengine-edge
journalctl -u visionengine-edge --since '2 minutes ago' --no-pager | grep 'fps (target' | tail -12
```

Expect two `enabled` lines and recent FPS reports for each camera. This tests
startup after restored power; it does not prove the cause of a previous outage
or guarantee recovery from every brownout or hardware fault.

---

## Diagnose an unexpected shutdown

These checks are read-only:

```bash
date
timedatectl
uptime
sudo journalctl --list-boots --no-pager
```

Select the boot that ended around the outage. Use its explicit boot ID rather
than relying on relative boot selection when timestamps appear inconsistent:

```bash
sudo journalctl -b <BOOT_ID> -o short-iso --no-pager -n 200
sudo journalctl -k -b <BOOT_ID> -o short-iso --no-pager -n 200
sudo ls -la /sys/fs/pstore
```

- A shutdown sequence indicates an orderly shutdown; investigate what requested it.
- An abrupt journal ending is consistent with power loss, a hard reset, or a hang.
  It does not by itself prove a wall-power failure.
- An empty crash store does not rule out a crash or hardware thermal protection.
- `hot-surface-alert` transitions alone do not prove a thermal shutdown. Check
  cooling, measured temperatures, and critical-trip evidence separately.
- An RTC reporting 1970 can cause misleading boot/service timestamps before
  network time synchronization. Check the RTC battery/configuration separately;
  it does not establish the cause of power loss.

If the board stays off after DC is restored, inspect the jumper and button
wiring, adapter, DC connector, and outlet. Auto-power-on is a carrier-board
function; changing Vision Engine's service cannot enable it.

---

## References

- [Yahboom aluminum case wiring guide](https://www.yahboom.net/study/Jetson-Metal)
- [NVIDIA carrier-board specification, section 3.4: J14](https://developer.nvidia.com/downloads/assets/embedded/secure/jetson/orin_nano/docs/jetson_orin_nano_devkit_carrier_board_specification_sp.pdf)
- [NVIDIA power-on and power-off guide](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/howto.html)

---

<div align="center">
<br/>

**[Documentation index](README.md)** · **[Deployment](DEPLOYMENT.md)** · **[Tools](TOOLS.md)** · **[Runtime stability](RUNTIME_STABILITY.md)**

<br/>
</div>
