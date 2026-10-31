#!/bin/bash
cd "$(dirname "$0")"
python3 download_sources.py
printf '\nEnter를 누르면 종료합니다: '
read -r _
