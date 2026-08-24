"""導納控制的安全機制：死區、飽和限幅、工作空間排斥場、力量閾值急停。

官方ros2_controllers的admittance_controller完全沒有這些機制(純數學核心)，
這塊參考nbfigueroa/robot_admittance_controller(ROS1, MIT IRG)的設計精神，
加上自訂的絕對力量閾值(參考Zhang et al. arXiv:2310.10509附錄)。
"""
import numpy as np


def apply_deadband(F_ext, deadband):
    """力/力矩死區：絕對值小於deadband的分量視為雜訊，歸零。

    deadband可以是純量(六軸共用同一個門檻)或6維陣列(各軸不同)。
    """
    F_ext = np.asarray(F_ext, dtype=float)
    deadband = np.asarray(deadband, dtype=float)
    return np.where(np.abs(F_ext) < deadband, 0.0, F_ext)


def saturate_vector(delta, max_magnitude):
    """向量整體飽和限幅：超過max_magnitude時等比例縮小，保留方向不失真。

    不能逐分量各自截斷——那樣會扭曲修正方向(例如xy各自截斷，會讓
    修正方向偏離原本合力的方向)，要整個向量一起等比例縮放。
    """
    delta = np.asarray(delta, dtype=float)
    norm = np.linalg.norm(delta)
    if norm > max_magnitude and norm > 1e-9:
        return delta * (max_magnitude / norm)
    return delta


def workspace_repulsion(xc, center, max_radius, gain):
    """工作空間排斥場：xc離center(3維位置)超過max_radius時，疊加一個
    指向center的修正向量，力度隨超出量線性增加；在範圍內回傳零向量，
    不影響原本的導納輸出。用途是避免導納把手臂帶出安全範圍。
    """
    xc = np.asarray(xc, dtype=float)
    center = np.asarray(center, dtype=float)
    offset = xc - center
    dist = np.linalg.norm(offset)
    if dist <= max_radius or dist < 1e-9:
        return np.zeros(3)
    direction_back = -offset / dist
    return direction_back * gain * (dist - max_radius)


def detect_force_rate_rise(F_now, F_prev, dt, rate_threshold):
    """偵測力訊號的變化率是否突然上升，用來判斷「碰到底」等接觸事件。

    參考US Patent 5940967(電子接頭插入)的做法：不是看絕對力量大小，是看
    「原本平穩、突然陡升」這個特徵——比固定絕對閾值更能適應不同插入速度
    /摩擦力造成的力量大小差異。F_now/F_prev建議用濾波後的訊號(降低雜訊
    誤觸發)，dt是兩次讀值間隔(秒)。
    """
    rate = (F_now - F_prev) / dt
    return bool(rate > rate_threshold)


def detect_force_rate_drop(F_now, F_prev, dt, rate_threshold):
    """偵測力訊號的變化率是否突然下降，用來判斷螺旋搜尋階段「掉進孔裡了」。

    跟detect_force_rate_rise方向相反：螺旋搜尋期間，靠虛擬穿透深度頂著
    接觸面，Z方向會維持一個穩定的下壓反作用力；一旦連接器滑進孔裡，
    接觸面消失，這個反作用力會在很短時間內驟降到接近0(不再需要那麼大
    力氣抵抗虛擬穿透量)，這個突降本身就是「找到孔了」的訊號，比持續看
    絕對力量大小可靠——不同姿態/連接器材質摩擦力不同，很難抓一個通用的
    絕對閾值，但「原本穩定施力、突然驟降」這個變化率特徵，不太受那些
    因素影響。

    rate_threshold: 呼叫端一律傳正數，內部自己處理正負號(跟
        detect_force_rate_rise統一傳正數的慣例一致，避免呼叫端自己記得
        加負號、容易搞混方向)，判斷邏輯是變化率比-rate_threshold還要負
        才觸發。
    """
    rate = (F_now - F_prev) / dt
    return bool(rate < -abs(rate_threshold))


def check_force_limit(F_ext, force_limit, torque_limit):
    """絕對力量閾值檢查，當作最後一道安全防線。

    F_ext是6維[Fx,Fy,Fz,Tx,Ty,Tz]。回傳True代表在安全範圍內，
    False代表超過閾值，呼叫端應該讓手臂停止/後退，不是繼續跑導納。
    """
    F_ext = np.asarray(F_ext, dtype=float)
    force_norm = np.linalg.norm(F_ext[0:3])
    torque_norm = np.linalg.norm(F_ext[3:6])
    return bool(force_norm <= force_limit and torque_norm <= torque_limit)
