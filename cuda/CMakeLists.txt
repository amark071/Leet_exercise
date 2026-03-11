cmake_minimum_required(VERSION 3.18)
project(VectorAdd LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# 查找CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# CUDA功能库
add_library(func_cuda STATIC func.cu)
set_target_properties(func_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# 定义5个可执行文件列表
set(EXECUTABLES 
    vector_add
)

# 为每个目标创建可执行文件
foreach(exec ${EXECUTABLES})
    add_executable(${exec} ${exec}.cpp)
    set_target_properties(${exec} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_link_libraries(${exec} func_cuda ${CUDA_LIBRARIES})
endforeach()

# ========== 运行测试并保存日志 ==========

# 创建日志目录（使用cmake -E确保目录存在）
set(LOG_DIR ${CMAKE_BINARY_DIR}/logs)
file(MAKE_DIRECTORY ${LOG_DIR})

# 获取当前时间戳
string(TIMESTAMP TIMESTAMP "%Y%m%d_%H%M%S")

# 创建运行脚本（使用tee同时输出到屏幕和文件）
file(WRITE ${CMAKE_BINARY_DIR}/run_all_script.sh "#!/bin/bash
set -e

cd \"$(dirname "$0")\"
LOG_DIR=\"${LOG_DIR}\"
TIMESTAMP=\"${TIMESTAMP}\"

# 确保日志目录存在
mkdir -p \"\${LOG_DIR}\"

ALL_LOG=\"\${LOG_DIR}/all_\${TIMESTAMP}.log\"
echo \"========================================\" | tee \"\${ALL_LOG}\"
echo \"  Running all executables\" | tee -a \"\${ALL_LOG}\"
echo \"  Time: \$(date)\" | tee -a \"\${ALL_LOG}\"
echo \"  Logs: \${LOG_DIR}\" | tee -a \"\${ALL_LOG}\"
echo \"========================================\" | tee -a \"\${ALL_LOG}\"
echo \"\" | tee -a \"\${ALL_LOG}\"

")

# 为每个程序添加运行命令
foreach(exec ${EXECUTABLES})
    file(APPEND ${CMAKE_BINARY_DIR}/run_all_script.sh "
echo \">>> [$(expr ${CMAKE_CURRENT_LIST_LINE} + 1)/5] Running ${exec}...\" | tee -a \"\${ALL_LOG}\"
echo \"---------- ${exec} output ----------\" | tee -a \"\${ALL_LOG}\"
if ${CMAKE_BINARY_DIR}/${exec} 2>&1 | tee -a \"\${ALL_LOG}\"; then
    echo \"[✓ SUCCESS] ${exec}\" | tee -a \"\${ALL_LOG}\"
else
    echo \"[✗ FAILED] ${exec} (exit code: \$?)\" | tee -a \"\${ALL_LOG}\"
fi
echo \"\" | tee -a \"\${ALL_LOG}\"
")
endforeach()

file(APPEND ${CMAKE_BINARY_DIR}/run_all_script.sh "
echo \"========================================\" | tee -a \"\${ALL_LOG}\"
echo \"  All executables completed\" | tee -a \"\${ALL_LOG}\"
echo \"  Time: \$(date)\" | tee -a \"\${ALL_LOG}\"
echo \"  Full log: \${ALL_LOG}\" | tee -a \"\${ALL_LOG}\"
echo \"========================================\" | tee -a \"\${ALL_LOG}\"
")

# 设置脚本权限
file(CHMOD ${CMAKE_BINARY_DIR}/run_all_script.sh 
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
)

# 创建运行目标
add_custom_target(run_all
    COMMAND /bin/bash ${CMAKE_BINARY_DIR}/run_all_script.sh
    DEPENDS ${EXECUTABLES}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all executables (output to screen + logs)..."
    VERBATIM
)

# ========== 单个程序运行（带日志） ==========

foreach(exec ${EXECUTABLES})
    set(SINGLE_LOG ${LOG_DIR}/${exec}_${TIMESTAMP}.log)
    add_custom_target(run_${exec}
        COMMAND /bin/bash -c "mkdir -p ${LOG_DIR} && ${CMAKE_BINARY_DIR}/${exec} 2>&1 | tee ${SINGLE_LOG}"
        DEPENDS ${exec}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running ${exec}..."
        VERBATIM
    )
endforeach()

# ========== 辅助目标 ==========

# 显示日志
add_custom_target(show_logs
    COMMAND ${CMAKE_COMMAND} -E echo "=== Log Files ==="
    COMMAND ls -lh ${LOG_DIR}/
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "=== Combined Log (last 50 lines) ==="
    COMMAND /bin/bash -c "tail -50 ${LOG_DIR}/all_*.log 2>/dev/null || echo 'No combined log found'"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# 清理日志
add_custom_target(clean_logs
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${LOG_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${LOG_DIR}
    COMMENT "Cleaning logs..."
)

# 一键测试
add_custom_target(test_all
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target run_all
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target show_logs
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)