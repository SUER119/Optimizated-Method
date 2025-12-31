import sys
sys.path.insert(0, "../../Python")
import Drivers
from JGSL import *

if __name__ == "__main__":
    sim = Drivers.FEMDiscreteShellBase("double", 3)

    # ----------------- 读取命令行参数 -----------------
    algI = 0
    if len(sys.argv) > 1:
        algI = int(sys.argv[1])

    clothI = 0
    if len(sys.argv) > 2:
        clothI = int(sys.argv[2])

    size = '85K'
    if len(sys.argv) > 3:
        size = sys.argv[3]
    
    membEMult = 0.01
    if len(sys.argv) > 4:
        membEMult = float(sys.argv[4])
    
    bendEMult = 0.1
    if len(sys.argv) > 5:
        bendEMult = float(sys.argv[5])

    # 摩擦系数（只剩下一块布，自碰撞）
    sim.mu = 0.4

    # =================================================
    # 场景：一块布，初始为水平方形片，从空中落下，
    #       只固定上边两个角点
    # =================================================

    # 把布放在 y = 0.3 的位置（可按需调高/调低）
    sim.add_shell_3D(
        "input/square" + size + ".obj",
        Vector3d(0.0, 0.3, 0.0),   # 平移：整体抬高
        Vector3d(0.0, 0.0, 0.0),   # 旋转中心（这里不用）
        Vector3d(0.0, 1.0, 0.0),   # 旋转轴
        0.0                        # 旋转角度
    )

    # 只剩 1 个物体，自碰撞摩擦矩阵是 1x1
    sim.muComp = StdVectorXd([sim.mu])

    # --------- 固定布的两个顶点（用 AABB 框选角点） ---------
    # 这里假设 square*.obj 在局部坐标大致是 [-0.5,0.5]^2，
    # 然后整体平移到 (0,0.3,0)。pin_width 可以适当调大/调小。
    pin_width = 0.02
    y_pin_min = 0.3 - 0.02
    y_pin_max = 0.3 + 0.02

    # 左上角点附近
    sim.set_DBC(
        Vector3d(-0.5 - pin_width, y_pin_min, -0.5 - pin_width),
        Vector3d(-0.5 + pin_width, y_pin_max, -0.5 + pin_width),
        Vector3d(0.0, 0.0, 0.0),   # 速度 = 0
        Vector3d(0.0, 0.0, 0.0),   # 旋转中心
        Vector3d(0.0, 1.0, 0.0),   # 旋转轴
        0.0                        # 角速度 = 0，完全固定
    )

    # 右上角点附近
    sim.set_DBC(
        Vector3d(0.5 - pin_width, y_pin_min, -0.5 - pin_width),
        Vector3d(0.5 + pin_width, y_pin_max, -0.5 + pin_width),
        Vector3d(0.0, 0.0, 0.0),
        Vector3d(0.0, 0.0, 0.0),
        Vector3d(0.0, 1.0, 0.0),
        0.0
    )

    # 不再需要旋转球、地面布片，相关 add_shell_3D 和 DBC 直接删除

    # ----------------- 时间步设置 -----------------
    sim.dt = 0.04
    sim.frame_dt = 0.04
    sim.frame_num = 100
    sim.withCollision = True   # 保留自碰撞

    # ----------------- 材料 + 算法选项（保持原始写法） -----------------
    if algI == 0:
        # iso
        sim.initialize(
            sim.cloth_density_iso[clothI],
            sim.cloth_Ebase_iso[clothI] * membEMult,
            sim.cloth_nubase_iso[clothI],
            sim.cloth_thickness_iso[clothI],
            0
        )
        sim.bendingStiffMult = bendEMult / membEMult
        sim.kappa_s = Vector2d(1e3, 0)
        sim.s = Vector2d(sim.cloth_SL_iso[clothI], 0)
    elif algI == 1:
        # iso, no strain limit
        sim.initialize(
            sim.cloth_density_iso[clothI],
            sim.cloth_Ebase_iso[clothI] * membEMult,
            sim.cloth_nubase_iso[clothI],
            sim.cloth_thickness_iso[clothI],
            0
        )
        sim.bendingStiffMult = bendEMult / membEMult
        sim.kappa_s = Vector2d(0, 0)
    elif algI == 2:
        # aniso
        sim.initialize(
            sim.cloth_density[clothI],
            sim.cloth_Ebase[clothI],  # 实际只影响弯曲
            0,
            sim.cloth_thickness[clothI],
            0
        )
        sim.bendingStiffMult = bendEMult
        sim.fiberStiffMult = sim.cloth_weftWarpMult[clothI] * membEMult
        sim.inextLimit = sim.cloth_inextLimit[clothI]
        sim.kappa_s = Vector2d(0, 0)

    # IPC 初始化 + 跑模拟
    sim.initialize_OIPC(1e-3, 0)
    sim.run()
