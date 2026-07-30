# 多模态无人机 Agent

这部分代码对应无人机巡检 Agent，核心是让多模态模型负责生成任务计划，由确定性规则决定动作能否执行。

## 核心链路

`图像/任务/遥测 -> 计划 JSON -> Schema -> Safety Gate -> 状态机 -> MAVSDK -> 遥测回读`

## 代码目录

- `src/drone_agent/contracts.py`：任务计划、动作白名单和遥测协议。
- `src/drone_agent/planner.py`：多模态 Planner 提示词与 JSON 解析。
- `src/drone_agent/safety.py`：遥测新鲜度、电量、定位、限高、地理围栏和人工批准。
- `src/drone_agent/state_machine.py`：任务状态转换与飞行状态恢复。
- `src/drone_agent/executor.py`：动作派发、超时处理、幂等和 Hold 恢复。
- `src/drone_agent/mavsdk_adapter.py`：MAVSDK Action API 适配。

## 关键设计

Planner 只输出 `takeoff/goto/inspect/hold/rtl/land` 六类高层动作。每一步执行前重新读取遥测并经过 Safety Gate；计划时间戳、遥测快照或状态机不一致时直接阻断。执行器记录 `step_id` 防止重复派发，适配器异常或超时后切换到失败状态并请求 Hold。
