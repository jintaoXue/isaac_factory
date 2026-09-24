#!/usr/bin/env bash
# DNS / 外网快速检查（5090 等机房机）
# 用法: bash ~/work/check_dns.sh

set -u
echo "==== host / time ===="
hostname
date

echo
echo "==== /etc/resolv.conf ===="
cat /etc/resolv.conf 2>/dev/null || echo "(missing)"

echo
echo "==== DNS resolve github.com ===="
if getent hosts github.com; then
  echo "OK: getent resolved github.com"
else
  echo "FAIL: getent could not resolve github.com"
fi
if command -v dig >/dev/null 2>&1; then
  dig +short github.com A 2>&1 | head -5
elif command -v nslookup >/dev/null 2>&1; then
  nslookup github.com 2>&1 | head -10
fi

echo
echo "==== IP connectivity (not DNS) ===="
if ping -c 2 -W 2 8.8.8.8 >/dev/null 2>&1; then
  echo "OK: ping 8.8.8.8"
else
  echo "FAIL: ping 8.8.8.8 (可能整机断外网)"
fi
if ping -c 2 -W 2 1.1.1.1 >/dev/null 2>&1; then
  echo "OK: ping 1.1.1.1"
else
  echo "FAIL: ping 1.1.1.1"
fi

echo
echo "==== DNS-dependent ping github.com ===="
if ping -c 2 -W 2 github.com >/dev/null 2>&1; then
  echo "OK: ping github.com"
else
  echo "FAIL: ping github.com (DNS 或外网问题)"
fi

echo
echo "==== HTTPS github.com ===="
if curl -sI --connect-timeout 5 https://github.com >/tmp/_dns_curl_hdr 2>/tmp/_dns_curl_err; then
  head -3 /tmp/_dns_curl_hdr
  echo "OK: curl https://github.com"
else
  echo "FAIL: curl https://github.com"
  cat /tmp/_dns_curl_err 2>/dev/null | head -5
fi

echo
echo "==== git ssh to github (optional) ===="
if timeout 8 ssh -o BatchMode=yes -o ConnectTimeout=5 -T git@github.com 2>&1 | head -5; then
  :
fi

echo
echo "==== default route ===="
ip route | head -5

echo
echo "==== tips ===="
cat <<'EOF'
若 getent/ping github.com 失败，但 8.8.8.8 通：多半是 DNS。可临时:
  sudo tee /etc/resolv.conf >/dev/null <<'EOR'
nameserver 8.8.8.8
nameserver 1.1.1.1
EOR
然后: getent hosts github.com && git pull

若 8.8.8.8 也不通：检查网线/Wi-Fi/网关，或:
  sudo systemctl restart NetworkManager
EOF
