#!/bin/sh
# Read-only version probe. Standalone Arena timeout: 10 seconds.
# Only documented SMI info and two SDK-defined RNGD sysfs attributes are read.
set -u

if ! command -v timeout >/dev/null 2>&1; then
    printf '%s\n' 'probe_unavailable: timeout command is absent; no queries run'
    exit 0
fi
if [ "${1-}" != '--bounded' ]; then
    timeout --kill-after=1s 8s sh "$0" --bounded
    probe_rc=$?
    if [ "$probe_rc" -ne 0 ]; then
        printf 'probe_incomplete: exit=%s; unavailable data does not imply incompatibility\n' "$probe_rc"
    fi
    exit "$probe_rc"
fi

printf '%s\n' 'arena_version_probe_v1' 'query: furiosa-smi info (default format)'
if command -v furiosa-smi >/dev/null 2>&1; then
    timeout --kill-after=1s 3s furiosa-smi info 2>&1
    probe_rc=$?
    printf 'furiosa_smi_info_exit=%s\n' "$probe_rc"
else
    printf '%s\n' 'furiosa_smi_info_unavailable: command not installed or not on PATH'
fi

probe_driver_count=0
probe_firmware_count=0
for probe_dir in /sys/class/rngd_mgmt/rngd!npu[0-9]*mgmt; do
    [ -d "$probe_dir" ] || continue
    for probe_attr in version fw_version; do
        probe_file="$probe_dir/$probe_attr"
        if [ ! -r "$probe_file" ]; then
            printf 'unavailable: %s (absent or unreadable)\n' "$probe_file"
            continue
        fi
        probe_value=$(timeout --kill-after=1s 1s head -c 512 "$probe_file" 2>/dev/null)
        probe_rc=$?
        if [ "$probe_rc" -ne 0 ] || [ -z "$probe_value" ]; then
            printf 'unavailable: %s (exit=%s or empty)\n' "$probe_file" "$probe_rc"
            continue
        fi
        printf '%s = %s\n' "$probe_file" "$probe_value"
        if [ "$probe_attr" = version ]; then
            probe_driver_count=$((probe_driver_count + 1))
        else
            probe_firmware_count=$((probe_firmware_count + 1))
        fi
    done
done

printf 'sysfs_driver_fields_read=%s sysfs_firmware_fields_read=%s\n' "$probe_driver_count" "$probe_firmware_count"
if [ "$probe_driver_count" -eq 0 ]; then
    printf '%s\n' 'driver_version_unavailable: SDK-defined sysfs attributes were not readable'
fi
printf '%s\n' 'probe_complete: observations only; no compatibility verdict'
