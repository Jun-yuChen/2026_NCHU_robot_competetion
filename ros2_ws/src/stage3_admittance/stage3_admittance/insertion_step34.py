"""Step3(對準滑入)/Step4(插入到底)的軌跡生成邏輯——純函式，不碰ROS2/
硬體狀態，風格比照admittance_core.py的linear_insertion_trajectory/
spiral_trajectory，方便獨立測試，之後再接進insertion_node_zero.py。

背景(2026-08-20設計討論)：Step2(Searching)結束時偵測到Fz驟降，代表接頭
尖端已經稍微滑進孔口邊緣，但還沒真正對準軸線、沒有真的往下插。

    Step3(對準滑入)：短距離延續下降(維持X/Y柔順，靠孔口導角把接頭被動
        導正到軸線上)，順便驗證Step2的驟降判定是不是誤判——如果繼續
        下降立刻又撞到力量上升，代表剛才是假訊號，不是真的滑進孔。
    Step4(插入到底)：對準後重新用linear_insertion_trajectory往下走剩餘
        深度，靠detect_force_rate_rise(safety.py，已存在但目前沒人用)
        偵測「力量突然上升」判斷插到底了，設計依據跟insertion_node.py
        的completion_rate_threshold/completion_min_travel_fraction同一
        套(參考US Patent 5940967：偵測力量突然上升而非絕對值)。
"""
import numpy as np


def aligning_trajectory(t, search_found_pose, align_depth, align_speed):
    """Step3對準滑入的參考軌跡：x,y,rx,ry,rz全程不動(維持Searching結束
    那一刻的位置，不主動橫向移動)，只有z軸等速小幅前進，讓X/Y方向的
    導納柔順自然發揮——孔口通常有導角，被動柔順下降時,接頭會被導角
    自然導正到軸線上，不需要主動規劃橫向修正路徑。

    跟linear_insertion_trajectory本質上是同一種軌跡形狀(z軸等速前進、
    其餘軸鎖定)，之所以獨立成一個函式而不是直接複用，是因為語意上
    這是「短距離對準滑入」，跟Step4「插入到底」是不同階段、不同終止
    條件(這裡固定走完align_depth就結束，不是靠力量判斷)，未來如果
    Step3要加其他行為(例如加入小幅度旋轉dither輔助對準)，改這裡不會
    牽動Step4的邏輯。

    t: 從進入Step3算起的經過時間(秒)
    search_found_pose: 6維，Step2偵測到驟降那一刻的xc(insertion frame下)
    align_depth: Step3要延續下降的總深度(公尺，沿z軸)，建議是個小值
        (例如2~3mm)，太大會失去「先驗證再插入」的用意
    align_speed: 下降速度(公尺/秒)，建議比Step4插入速度更慢，因為這段
        本質上是在探測「有沒有真的滑進去」，慢一點比較安全

    回傳：(xd, xd_dot, xd_ddot, done)，前三個皆為6維向量，done是bool，
        True代表已經走完align_depth(可以判斷Step3是否成功、要不要進
        Step4)
    """
    search_found_pose = np.asarray(search_found_pose, dtype=float)
    xd = search_found_pose.copy()
    xd_dot = np.zeros(6)
    xd_ddot = np.zeros(6)

    z_travelled = align_speed * t
    done = z_travelled >= align_depth

    if done:
        xd[2] = search_found_pose[2] + align_depth
    else:
        xd[2] = search_found_pose[2] + z_travelled
        xd_dot[2] = align_speed

    return xd, xd_dot, xd_ddot, done


def check_step3_false_positive(F_ext_z, reject_force_threshold):
    """檢查Step3延續下降過程中，力量有沒有再度飆升——如果Step2的驟降
    判定其實是假訊號(尖端沒有真的滑進孔，只是雜訊或短暫脫離接觸)，
    Step3繼續下降時會立刻再撞到阻力；如果是真的滑進孔，這段下降應該
    要維持在低力量(死區量級)才對。

    F_ext_z: 當下濾波後的Z軸力讀值(N)，建議傳入死區處理"前"的濾波值
        (跟Searching找孔判定用同一個訊號來源F_z_current_filtered)，
        不要用死區處理後的值——死區可能把小幅上升的力吃掉，延誤偵測
    reject_force_threshold: 判定為「又撞到了、Step2結果不可信」的力量
        閾值(N)，建議跟contact_force_threshold同量級或略高，不要用
        early_contact_force_threshold那麼高的值(那是給Approaching的
        異常判定用，這裡的情境不同)

    回傳：bool，True代表偵測到疑似誤判，Step3應該中止、Step2判定不可信
    """
    return abs(F_ext_z) > reject_force_threshold
