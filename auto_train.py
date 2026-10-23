import subprocess
import sys


def run_phase(command_list):
    cmd_str = " ".join(command_list)
    print(f"\n{'=' * 60}")
    print(f"🚀 开始自动执行: {cmd_str}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(command_list)

    # Terminate pipeline if stage fails
    if result.returncode != 0:
        print(f"\n❌ 训练中断！命令执行失败: {cmd_str}")
        sys.exit(result.returncode)

    print(f"\n✅ 阶段完成: {cmd_str}\n")


if __name__ == "__main__":
    # Pipeline stages
    phase1_cmd = ["python", "train.py", "--phase", "1"]
    phase2_cmd = ["python", "train.py", "--phase", "2", "--resume", "outputs/checkpoints/phase1_best.pth"]
    phase3_cmd = ["python", "train.py", "--phase", "3", "--resume", "outputs/checkpoints/phase2_best.pth"]

    run_phase(phase1_cmd)
    run_phase(phase2_cmd)
    run_phase(phase3_cmd)

    print("🎉 所有阶段训练完毕！")
