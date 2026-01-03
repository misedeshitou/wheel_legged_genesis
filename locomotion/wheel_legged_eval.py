import argparse
import copy
import os
import pickle
import sys

import genesis as gs
import numpy as np
import torch

# 处理路径，确保能导入 utils 和环境文件
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import gamepad

# 初始化 Genesis
gs.init(backend=gs.gpu)

# 导入依赖 Genesis 的模块（确保在 gs.init 之后）
from rsl_rl.runners import OnPolicyRunner
from wheel_legged_env import WheelLeggedEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--exp_name", type=str, default="wheel-legged-walking-v0.3.0"
    )
    parser.add_argument("--ckpt", type=int, default=7300)
    args = parser.parse_args()

    log_dir = f"logs/{args.exp_name}"
    cfgs_path = os.path.join(log_dir, "cfgs.pkl")

    if not os.path.exists(cfgs_path):
        print(f"错误: 找不到配置文件 {cfgs_path}")
        return

    # 加载配置
    (
        env_cfg,
        obs_cfg,
        reward_cfg,
        command_cfg,
        curriculum_cfg,
        domain_rand_cfg,
        terrain_cfg,
        train_cfg,
    ) = pickle.load(open(cfgs_path, "rb"))

    # --- 关键修改：评估模式环境调整 ---
    terrain_cfg["terrain"] = True
    terrain_cfg["eval"] = "agent_eval_gym"

    # 强制关闭领域随机化，避免 Genesis 0.3.x 在单环境下尝试更新动力学参数时报错
    domain_rand_cfg["randomize_rigids_prop"] = False
    domain_rand_cfg["push_robots"] = False

    # 初始化环境
    env = WheelLeggedEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        curriculum_cfg=curriculum_cfg,
        domain_rand_cfg=domain_rand_cfg,
        terrain_cfg=terrain_cfg,
        robot_morphs="urdf",
        show_viewer=True,
        num_view=1,
        train_mode=False,  # 设置为评估模式
    )

    # 初始化训练器并加载模型
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")

    if os.path.exists(resume_path):
        runner.load(resume_path)
        print(f"成功加载检查点: {resume_path}")
    else:
        print(f"警告: 找不到检查点 {resume_path}")
        return

    # JIT 模型导出逻辑
    print("\n--- 正在导出并加载 JIT 策略 ---")
    try:
        # 将 actor 复制并转到 CPU 进行脚本化
        actor_model = copy.deepcopy(runner.alg.actor_critic.actor).to("cpu")
        jit_policy_path = os.path.join(log_dir, "policy.pt")
        torch.jit.script(actor_model).save(jit_policy_path)

        # 重新加载用于测试
        loaded_policy = torch.jit.load(jit_policy_path)
        loaded_policy.eval()
        loaded_policy.to("cuda:0")
        print("JIT 模型导出并加载成功!")
    except Exception as e:
        print(f"JIT 导出失败: {e}")
        # 如果 JIT 失败，回退到原始推理策略
        loaded_policy = runner.get_inference_policy(device="cuda:0")

    # 重置环境
    obs, _ = env.reset()

    # 手柄控制初始化
    # 参数: [lin_x, lin_y, ang_z, leg_l, leg_r, tsk]
    pad = gamepad.control_gamepad(command_cfg, [1.2, 0.0, 10.0, 0.05, 0.05, 1.0])

    print("\n--- 开始评估 (按手柄或键盘重置键可重置) ---")
    with torch.no_grad():
        while True:
            # 执行策略推理
            actions = loaded_policy(obs)
            obs, _, rews, dones, infos = env.step(actions)

            # 获取手柄指令并下发
            commands, reset_flag = pad.get_commands()
            env.set_commands(np.arange(env.num_envs), commands)

            # 手动重置逻辑
            if reset_flag:
                env.reset()


if __name__ == "__main__":
    main()
