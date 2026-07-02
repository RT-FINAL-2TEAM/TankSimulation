#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ -n "${ANDROID_HOME:-}" && -d "${ANDROID_HOME}" ]]; then
  SDK="$ANDROID_HOME"
elif [[ -n "${ANDROID_SDK_ROOT:-}" && -d "${ANDROID_SDK_ROOT}" ]]; then
  SDK="$ANDROID_SDK_ROOT"
elif [[ -d "$HOME/Android/Sdk" ]]; then
  SDK="$HOME/Android/Sdk"
elif [[ -d "$HOME/.android/sdk" ]]; then
  SDK="$HOME/.android/sdk"
elif [[ -d "/opt/android-sdk" ]]; then
  SDK="/opt/android-sdk"
else
  echo "Android SDK directory was not found automatically."
  echo "Open Android Studio > Settings > Languages & Frameworks > Android SDK and check the SDK path."
  echo "Then create local.properties manually, for example:"
  echo "  echo 'sdk.dir=/home/tankcc/Android/Sdk' > local.properties"
  exit 1
fi

printf 'sdk.dir=%s\n' "$SDK" > local.properties
echo "Wrote local.properties with sdk.dir=$SDK"
