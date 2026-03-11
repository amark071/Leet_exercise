#!/bin/bash
set -e

cd "$(dirname $0)"
LOG_DIR="/home/master/Leet_exercise/cuda/build/logs"
TIMESTAMP="20260311_111254"

# 确保日志目录存在
mkdir -p "${LOG_DIR}"

ALL_LOG="${LOG_DIR}/all_${TIMESTAMP}.log"
echo "========================================" | tee "${ALL_LOG}"
echo "  Running all executables" | tee -a "${ALL_LOG}"
echo "  Time: $(date)" | tee -a "${ALL_LOG}"
echo "  Logs: ${LOG_DIR}" | tee -a "${ALL_LOG}"
echo "========================================" | tee -a "${ALL_LOG}"
echo "" | tee -a "${ALL_LOG}"


echo ">>> [$(expr 66 + 1)/5] Running vector_add..." | tee -a "${ALL_LOG}"
echo "---------- vector_add output ----------" | tee -a "${ALL_LOG}"
if /home/master/Leet_exercise/cuda/build/vector_add 2>&1 | tee -a "${ALL_LOG}"; then
    echo "[✓ SUCCESS] vector_add" | tee -a "${ALL_LOG}"
else
    echo "[✗ FAILED] vector_add (exit code: $?)" | tee -a "${ALL_LOG}"
fi
echo "" | tee -a "${ALL_LOG}"

echo "========================================" | tee -a "${ALL_LOG}"
echo "  All executables completed" | tee -a "${ALL_LOG}"
echo "  Time: $(date)" | tee -a "${ALL_LOG}"
echo "  Full log: ${ALL_LOG}" | tee -a "${ALL_LOG}"
echo "========================================" | tee -a "${ALL_LOG}"
