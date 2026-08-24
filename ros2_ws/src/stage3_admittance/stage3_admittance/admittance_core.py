"""導納控制的純數學核心，不碰ROS2/硬體，方便單元測試與未來RL擴充。

六維向量的軸序統一為 [x, y, z, rx, ry, rz]。
演算法核心參考官方 ros-controls/ros2_controllers 的 admittance_controller：
https://control.ros.org/rolling/doc/ros2_controllers/admittance_controller/doc/userdoc.html
"""
import numpy as np

GRAVITY_DIR_WORLD = np.array([0.0, 0.0, -1.0])


def euler_to_matrix(rx, ry, rz):
    """歐拉角轉旋轉矩陣，假設ZYX慣例(R = Rz(rz) @ Ry(ry) @ Rx(rx))。

    ⚠️ 這個慣例是常見假設，不是確認過的TM官方定義，等真機到手要驗證：
    轉手臂繞單一已知軸轉已知角度，比對算出來的重力方向跟實際量到的
    方向是否一致，不一致就要換慣例重推。
    """
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def compensate_gravity(F_raw, R_sensor_to_world, cog_pos, cog_force, bias=None):
    """扣掉末端工具重量在當前姿態下、於感測器座標系造成的力/力矩分量，
    以及感測器本身的固定偏移(bias)。

    重力在世界座標系永遠朝下，但感測器隨手臂姿態轉動，所以「重力看起來
    像多少力」每個姿態都不一樣，不能只靠開機歸零一次就固定不變。

    bias是感測器讀值裡跟姿態/外力都無關的固定偏移量(gravity_calib.py的
    solve_mass_cog_bias()有解出這一項)，這是2026-08-12真機測試踩到的坑：
    一開始沒有扣這個bias，剛啟動導納迴圈(還沒受到任何外力)Fz就量到
    十幾牛頓的殘留值，直接誤觸發「插入初期力量異常」保護機制——bias
    不扣的話，不管重力補償算得多準，殘留的固定偏移還是會讓F_ext失真。

    F_raw: 原始六軸讀值 [Fx,Fy,Fz,Tx,Ty,Tz]，感測器座標系
    R_sensor_to_world: 3x3旋轉矩陣，感測器座標系 -> 世界(base)座標系
    cog_pos: 工具重心位置(3維)，相對感測器原點，感測器座標系下的座標(公尺)
    cog_force: 工具重量(N)
    bias: 6維，感測器固定偏移量(跟cog_force同單位)，None代表不扣(相容舊呼叫)
    回傳：F_ext，扣除重力分量+bias後的六軸讀值
    """
    F_raw = np.asarray(F_raw, dtype=float)
    cog_pos = np.asarray(cog_pos, dtype=float)

    F_grav_sensor = R_sensor_to_world.T @ (GRAVITY_DIR_WORLD * cog_force)
    T_grav_sensor = np.cross(cog_pos, F_grav_sensor)

    F_ext = F_raw.copy()
    F_ext[0:3] -= F_grav_sensor
    F_ext[3:6] -= T_grav_sensor
    if bias is not None:
        F_ext -= np.asarray(bias, dtype=float)
    return F_ext


def filter_wrench(F_ext, F_ext_prev, alpha):
    """一階指數平滑濾波：F_filtered = alpha*F_ext + (1-alpha)*F_ext_prev。

    alpha越小濾得越用力(訊號越平滑但延遲越大)，官方預設0.05。
    """
    F_ext = np.asarray(F_ext, dtype=float)
    F_ext_prev = np.asarray(F_ext_prev, dtype=float)
    return alpha * F_ext + (1.0 - alpha) * F_ext_prev


def damping_from_ratio(M, K, zeta):
    """D = zeta * 2*sqrt(M*K)，官方的阻尼比參數化公式。

    zeta=1代表臨界阻尼(最快收斂不震盪)，是保守起始值的建議選擇。
    """
    M = np.asarray(M, dtype=float)
    K = np.asarray(K, dtype=float)
    zeta = np.asarray(zeta, dtype=float)
    return zeta * 2.0 * np.sqrt(M * K)


def admittance_step(F_ext, xc, xc_dot, xd, xd_dot, xd_ddot, M, K, D, dt, selected_axes, F_desired=None):
    """核心導納動力學：M(ẍc-ẍd)+D(ẋc-ẋd)+K(xc-xd)=F_ext-F_desired，半隱式尤拉積分。

    每個週期呼叫一次，xc/xc_dot是上一週期算出的柔順位置/速度，回傳這個
    週期的新值。selected_axes是6個bool，False的軸不跑導納、直接等於
    參考軌跡(維持原本規劃、不修正)——目前只開x,y,z平移三軸，旋轉軸
    先關閉，交給前面的治具轉正步驟處理姿態誤差。

    2026-08-19新增F_desired(期望力，預設None=零向量，不影響原本呼叫方
    insertion_node.py/admittance_test_node.py的行為)：參考Delta6論文
    (arXiv:2604.06150)描述的peg-in-hole搜尋階段設計——不是用固定的
    虛擬穿透深度(xd往下偏移一個猜測值)去換算貼壓力，而是直接在動力學
    方程式裡加一個期望力項，平衡點(e_ddot=0、e_dot=0)會落在F_ext=
    F_desired的地方，系統自動收斂到「感測器讀到剛好等於期望壓力」的
    深度，不需要事先知道真實表面在哪裡，解決virtual penetration depth
    需要反覆猜測、猜不準就會懸空或衝過頭的問題。

    回傳：(xc_new, xc_dot_new)，皆為6維向量
    """
    F_ext = np.asarray(F_ext, dtype=float)
    xc = np.asarray(xc, dtype=float)
    xc_dot = np.asarray(xc_dot, dtype=float)
    xd = np.asarray(xd, dtype=float)
    xd_dot = np.asarray(xd_dot, dtype=float)
    xd_ddot = np.asarray(xd_ddot, dtype=float)
    M = np.asarray(M, dtype=float)
    K = np.asarray(K, dtype=float)
    D = np.asarray(D, dtype=float)
    selected_axes = np.asarray(selected_axes, dtype=bool)
    F_desired = np.zeros(6) if F_desired is None else np.asarray(F_desired, dtype=float)

    e = xc - xd
    e_dot = xc_dot - xd_dot
    e_ddot = (F_ext - F_desired - D * e_dot - K * e) / M
    xc_ddot = xd_ddot + e_ddot

    xc_dot_new = xc_dot + xc_ddot * dt
    xc_new = xc + xc_dot_new * dt

    # 沒開的軸直接鎖回參考軌跡，不累積導納偏移
    xc_new = np.where(selected_axes, xc_new, xd)
    xc_dot_new = np.where(selected_axes, xc_dot_new, xd_dot)

    return xc_new, xc_dot_new


def spiral_trajectory(t, contact_pose, penetration_depth, pitch, max_radius, angular_speed):
    """螺旋搜尋參考軌跡：XY走阿基米德螺旋(半徑隨時間線性變大)，Z固定在
    一個「虛擬穿透深度」，靠admittance_step本身的K彈簧力自然貼壓在接觸
    面上，不需要另外寫一套主動力控制迴圈——這是"方案A"的核心技巧：
    把xd往下設定超過實際碰到的表面，K會產生反作用力頂著這個虛擬深度，
    近似維持一個固定下壓力(實際力量大小≈K_z*penetration_depth，會再被
    safety.saturate_vector限幅，真正拿到的力上限是K_z*max_step_correction，
    調penetration_depth前記得對照當下的K_z/max_step_correction，不然
    虛擬深度設太深也只會被限幅頂住，不會產生更大的力)。

    阿基米德螺旋：r(θ)=pitch/(2π)·θ，θ隨時間等速增加(angular_speed)，
    相鄰兩圈間距固定是pitch，涵蓋範圍均勻不重疊，比等速率半徑增加更
    容易控制搜尋密度。半徑超過max_radius後停止擴大(維持在最外圈打轉)，
    呼叫端要自己判斷「已經到max_radius還沒找到」代表搜尋失敗。

    t: 從進入搜尋階段算起的經過時間(秒)
    contact_pose: 6維，Step1偵測到碰觸那一刻的xc(insertion frame下)，
        螺旋中心點+Z參考基準+旋轉軸參考值(旋轉軸原封不動照抄，不參與
        搜尋運算)
    penetration_depth: 虛擬穿透深度(公尺)，必須是正值
    pitch: 螺旋每圈半徑增加量(公尺)
    max_radius: 螺旋最大搜尋半徑(公尺)
    angular_speed: 螺旋角速度(rad/s)

    回傳：(xd, xd_dot, xd_ddot)，皆為6維向量。
    """
    contact_pose = np.asarray(contact_pose, dtype=float)
    theta = angular_speed * t
    r_unclamped = (pitch / (2.0 * np.pi)) * theta
    r = min(r_unclamped, max_radius)

    xd = contact_pose.copy()
    xd_dot = np.zeros(6)
    xd_ddot = np.zeros(6)

    xd[0] = contact_pose[0] + r * np.cos(theta)
    xd[1] = contact_pose[1] + r * np.sin(theta)
    xd[2] = contact_pose[2] - penetration_depth

    if r_unclamped < max_radius:
        # r本身也是t的函數(r=k*angular_speed*t，k=pitch/2π)，跟theta一起
        # 對t微分要用連鎖律/乘積法則展開，不能只把r當常數處理。
        r_dot = (pitch / (2.0 * np.pi)) * angular_speed
        xd_dot[0] = r_dot * np.cos(theta) - r * angular_speed * np.sin(theta)
        xd_dot[1] = r_dot * np.sin(theta) + r * angular_speed * np.cos(theta)
        xd_ddot[0] = (
            -2.0 * r_dot * angular_speed * np.sin(theta) - r * angular_speed ** 2 * np.cos(theta)
        )
        xd_ddot[1] = (
            2.0 * r_dot * angular_speed * np.cos(theta) - r * angular_speed ** 2 * np.sin(theta)
        )
    # r已經到max_radius(停在最外圈打轉)時，速度/加速度維持0，不再持續繞圈
    # 累積角度——避免xd_dot/xd_ddot無限增長，且呼叫端本來就該在這個情況
    # 判斷搜尋失敗、不會真的一直跑下去。

    return xd, xd_dot, xd_ddot


def blend_velocity(v_from, v_to, t, blend_duration):
    """線性混合兩個速度向量，t=0時等於v_from，t>=blend_duration後完全
    等於v_to，中間線性內插——用於相位切換瞬間避免速度不連續造成機器人
    頓挫(例如Approaching結束時手臂還帶著穩定下降速度，但下一階段的
    自然速度接近0，若沒有過渡，PVT指令會要求瞬間變速)。

    v_from: 切換那一刻的速度(6維)
    v_to: 新階段自然算出來的速度(6維)
    t: 進入新階段後經過的時間(秒)
    blend_duration: 混合過渡時間(秒)，<=0代表不過渡、直接回傳v_to

    回傳：混合後的速度(6維)。
    """
    v_from = np.asarray(v_from, dtype=float)
    v_to = np.asarray(v_to, dtype=float)
    if blend_duration <= 0:
        return v_to.copy()
    frac = np.clip(t / blend_duration, 0.0, 1.0)
    return (1.0 - frac) * v_from + frac * v_to


def force_hold_z_velocity(F_ext_z, F_desired_z, gain, max_speed):
    """簡單比例力控制，回傳Z軸下壓速度(m/s)——不是導納(沒有M/K/D動態)，
    刻意設計成這樣：insertion_node_zero.py的Searching階段曾經用F_desired
    admittance機制維持貼壓力，實測發現它的收斂動態(約1秒時間常數)跟
    螺旋搜尋的旋轉週期(約3.14秒)搭不上，兩者互相干擾造成correction持續
    小幅震盪不收斂(見2026-08-20的調參紀錄)。這裡改用最簡單的比例控制，
    沒有二階動態、沒有跟旋轉週期共振的可能性：力不夠(F_ext_z跟F_desired_z
    差距為正)就往下壓，力夠了差距趨近0速度也趨近0，力超過(差距為負)
    就往上抬一點，全程用max_speed限幅避免單一tick位移過大。

    F_ext_z: 當下濾波後的Z軸力讀值(N)
    F_desired_z: 想維持的貼壓力(N)，正值代表希望感測到多少壓力
    gain: 比例增益(m/s per N)，越大反應越快但越容易震盪
    max_speed: 速度限幅(m/s)，避免單一control tick移動過大

    回傳：z軸下壓速度(m/s)，正值代表往下(插入方向)。
    """
    z_dot = gain * (F_desired_z - F_ext_z)
    return float(np.clip(z_dot, -max_speed, max_speed))


def linear_insertion_trajectory(t, start_pose, insertion_depth, insertion_speed):
    """參考軌跡產生器：x,y,rx,ry,rz全程不動，z軸等速前進插入。

    對應Stage1設計「直線接近+導納柔順插入」——導納只負責在這條軌跡上疊加
    修正量，不是取代它。假設運算是在TCP座標系下做，z軸對齊插入方向
    (跟你們CAD規範「+Z朝插入方向」一致)，start_pose是開始插入瞬間的姿態。

    t: 從開始插入算起的經過時間(秒)
    start_pose: 開始插入瞬間的6維姿態[x,y,z,rx,ry,rz]
    insertion_depth: 總插入深度(公尺，沿z軸)
    insertion_speed: 插入速度(公尺/秒)

    回傳：(xd, xd_dot, xd_ddot)，皆為6維向量。到達目標深度後z軸停住不再前進。
    """
    start_pose = np.asarray(start_pose, dtype=float)
    xd = start_pose.copy()
    xd_dot = np.zeros(6)
    xd_ddot = np.zeros(6)

    z_travelled = insertion_speed * t
    if z_travelled >= insertion_depth:
        xd[2] = start_pose[2] + insertion_depth
    else:
        xd[2] = start_pose[2] + z_travelled
        xd_dot[2] = insertion_speed

    return xd, xd_dot, xd_ddot
