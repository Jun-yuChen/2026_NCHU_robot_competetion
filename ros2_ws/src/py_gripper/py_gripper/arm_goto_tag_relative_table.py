#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
arm_goto_fixed_table.py 的 AprilTag 相對版本——完全獨立的新檔案，
不改動原本的 arm_goto_fixed_table.py。

差異：原本三個抓取點(含預備點)是寫死的絕對座標，工作台一旦被移動
就全部要重新量測。這支改成：工作台上固定貼一顆共用的 AprilTag，
三個抓取點的位置改成「相對於這顆 tag 的固定偏移」，每次抓取前先
偵測 tag 現在在哪，再換算出當下三個點實際的絕對座標——工作台移動
後不用重新量測，只要 tag 跟工作台的相對位置沒變就好。

主機前方的回位點(HOST_FRONT_POSE_M)維持寫死絕對座標，不隨 tag 變動
（主機/電腦位置不會跟著工作台一起移動）。

依賴同目錄的 apriltag_three_view_teach.py（跟 precision_cable_grip_multi_hole.py
共用同一套偵測/移動框架）。

示教流程（每個點只需做一次，工作台之後移動不用重教）：
  1. 把手臂 jog 到看得到 tag 的地方（不用是抓取點），執行：
       snapshot_tag --point-id 1
  2. 把手臂 jog 到點 1 真正的抓取位置，執行：
       teach_point_from_snapshot --point-id 1
     （如果抓取點本身就看得到 tag，也可以直接用 teach_point，
      跳過 snapshot 這一步。）
  3. 預備點會自動用「抓取點沿 base X 退後 100mm」算出來，不用另外教。
  4. 點 2、點 3 依此類推。

執行：
  run_pick --point-id 1        跑單一點的完整抓取流程
  run_pick_all --point-ids 1,2,3   依序跑三個點
"""

import argparse
import time
from pathlib import Path

import numpy as np
import rclpy

from tm_msgs.srv import SetIO

from apriltag_three_view_teach import (
    AprilTagTeachNode,
    invert_T,
    load_matrix,
    pose_to_T,
    T_to_pose,
)


PANEL_DIR = Path(__file__).resolve().parent
DATA_DIR = PANEL_DIR / "data"

IMAGE_TOPIC = "/camera/camera/color/image_raw"
INFO_TOPIC = "/camera/camera/color/camera_info"

# 主機前方固定放置點（不隨 tag 變動，維持絕對座標）。
# 原始量測 (robot base)：x=371.82, y=-155.44, z=423.01 (mm)
#                        rx=93.88, ry=0.89, rz=88.02 (deg)
HOST_FRONT_POSE_M = [0.37182, -0.15544, 0.42301, 1.638515, 0.015533, 1.536239]

RETREAT_DISTANCE_M = 0.10       # 抓取後沿 base X 退出的距離，跟預備點退後方向一致
ARRIVE_POS_ERROR_M = 0.005
ARRIVE_ROT_ERROR_DEG = 3.0

GRIPPER_OPEN_STATE = 0.0
GRIPPER_CLOSE_STATE = 1.0
GRIPPER_ACTUATION_DELAY_S = 1.0

# tm_driver 內部用 vel/pi 無條件捨去成整數百分比，太小的值會直接變成 0%
# （完全不動）。這兩個值沿用 arm_goto_fixed_table.py 已驗證過能動的設定。
TRANSIT_VELOCITY = 0.35   # PTP_T 長距離移動（預備點、回主機前方）
PRECISION_VELOCITY = 0.15  # LINE_T 短距離精準移動（預備點→抓取點、後退）


def config_file_for(point_id):
    suffix = f"_{point_id}" if point_id else ""
    return DATA_DIR / f"arm_goto_tag_relative_teach{suffix}.yaml"


def load_config(point_id=""):
    import yaml
    config_file = config_file_for(point_id)
    if not config_file.is_file():
        return {}
    with config_file.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def save_config(data, point_id=""):
    import yaml
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with config_file_for(point_id).open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def make_node(tag_size, tag_id):
    return AprilTagTeachNode(
        tag_size=float(tag_size),
        tag_id=int(tag_id),
        image_topic=IMAGE_TOPIC,
        info_topic=INFO_TOPIC,
    )


def wait_feedback(node, need_camera=True):
    node.spin_until_ready(
        need_camera=need_camera,
        need_joints=True,
        timeout=12.0,
    )


def current_tool_pose(node):
    return np.asarray(
        [float(v) for v in node.current_tool_pose[:6]],
        dtype=np.float64,
    )


def detect_base_tag(node, T_G_C, samples):
    """Detect the table AprilTag once (stable multi-frame estimate)."""
    wait_feedback(node, need_camera=True)
    T_C_Tag = node.detect_tag_stable(
        count=int(samples),
        timeout=max(6.0, 1.5 * int(samples)),
    )
    T_B_G = pose_to_T(current_tool_pose(node))
    T_B_Tag = T_B_G @ T_G_C @ T_C_Tag
    return T_B_Tag, T_C_Tag, T_B_G


def _load_eye(args):
    return load_matrix(args.eye_yaml, args.eye_key, invert=args.eye_invert)


def send_pose(node, pose, velocity, label, motion, timeout=30.0):
    pose = np.asarray(pose, dtype=np.float64).reshape(6)
    print("\n----------------------------------------")
    print(label)
    print("----------------------------------------")
    print(np.array2string(pose, precision=8, separator=", "))
    node.send_pose(
        pose.tolist(),
        velocity=float(velocity),
        acc_time=0.5,
        motion=motion,
    )
    node.wait_arrived(
        pose,
        pos_error=ARRIVE_POS_ERROR_M,
        rot_error_deg=ARRIVE_ROT_ERROR_DEG,
        timeout=float(timeout),
    )


def send_gripper(node, close):
    label = "CLOSE" if close else "OPEN"
    state = 1.0 if close else 0.0

    cli = node.create_client(SetIO, "set_io")
    if not cli.wait_for_service(timeout_sec=5.0):
        raise RuntimeError("找不到 /set_io service；請確認 TM Driver / Listen 已連線。")

    if not hasattr(SetIO.Request, "MODULE_ENDEFFECTOR"):
        raise RuntimeError("tm_msgs/SetIO 缺少 MODULE_ENDEFFECTOR")
    if not hasattr(SetIO.Request, "TYPE_DIGITAL_OUT"):
        raise RuntimeError("tm_msgs/SetIO 缺少 TYPE_DIGITAL_OUT")

    req = SetIO.Request()
    req.module = SetIO.Request.MODULE_ENDEFFECTOR
    req.type = SetIO.Request.TYPE_DIGITAL_OUT
    req.pin = 0
    req.state = float(state)

    print(f"\nGRIPPER {label}: ENDEFFECTOR DO_0 state={state:.1f}")

    future = cli.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)

    if not future.done() or future.result() is None:
        raise RuntimeError(f"夾爪 {label} /set_io 呼叫失敗")

    result = future.result()
    if hasattr(result, "ok") and not bool(result.ok):
        raise RuntimeError(f"夾爪 {label} /set_io rejected: {result}")


def snapshot_tag(args):
    """在看得到 tag 的地方拍一次，記住 tag 現在在哪（不用是抓取點）。"""
    T_G_C = _load_eye(args)
    rclpy.init()
    node = make_node(args.tag_size, args.tag_id)
    try:
        T_B_Tag, T_C_Tag, _ = detect_base_tag(node, T_G_C, args.samples)

        data = load_config(args.point_id)
        data.update({
            "point_id": args.point_id,
            "tag_id": int(args.tag_id),
            "tag_size_m": float(args.tag_size),
            "tag_snapshot_saved_at_unix": float(time.time()),
            "tag_snapshot_T_Base_AprilTag": T_B_Tag.tolist(),
        })
        save_config(data, args.point_id)

        print("\n========================================")
        print(f"Tag 快照已記錄（點位: {args.point_id or '(預設)'}）")
        print("========================================")
        print("T_Base_AprilTag:")
        print(np.array2string(T_B_Tag, precision=8, separator=", "))
        print("接下來把手臂移到真正的抓取點，執行 teach_point_from_snapshot。")
        print("saved:", config_file_for(args.point_id))
        print("========================================")
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


def _save_point_teach(data, point_id, T_Tag_Grab):
    """算出並存檔抓取點 + 自動退後 100mm 算出的預備點（相對 tag 的偏移）。"""
    # 預備點姿態跟抓取點完全相同，只有 base X 退後 RETREAT_DISTANCE_M——
    # 這一步是純矩陣運算，不需要額外示教或偵測。
    T_Tag_Approach = T_Tag_Grab.copy()
    T_Tag_Approach[0, 3] -= RETREAT_DISTANCE_M

    data.update({
        "point_id": point_id,
        "teach_saved_at_unix": float(time.time()),
        "T_Tag_GrabPoint": T_Tag_Grab.tolist(),
        "T_Tag_ApproachPoint": T_Tag_Approach.tolist(),
        "retreat_distance_m": RETREAT_DISTANCE_M,
    })
    save_config(data, point_id)
    return T_Tag_Approach


def teach_point(args):
    """直接在抓取點示教（抓取點本身看得到 tag 時用這個）。"""
    T_G_C = _load_eye(args)
    rclpy.init()
    node = make_node(args.tag_size, args.tag_id)
    try:
        T_B_Tag, _, T_B_G = detect_base_tag(node, T_G_C, args.samples)
        T_Tag_Grab = invert_T(T_B_Tag) @ T_B_G

        data = load_config(args.point_id)
        T_Tag_Approach = _save_point_teach(data, args.point_id, T_Tag_Grab)

        print("\n========================================")
        print(f"點位 {args.point_id or '(預設)'} 示教完成")
        print("========================================")
        print("T_Tag_GrabPoint:")
        print(np.array2string(T_Tag_Grab, precision=8, separator=", "))
        print("T_Tag_ApproachPoint (自動退後 100mm 算出):")
        print(np.array2string(T_Tag_Approach, precision=8, separator=", "))
        print("saved:", config_file_for(args.point_id))
        print("========================================")
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


def teach_point_from_snapshot(args):
    """用之前 snapshot_tag 記錄的 tag 位置示教（抓取點本身看不到 tag 時用這個）。"""
    data = load_config(args.point_id)
    raw_T_B_Tag = data.get("tag_snapshot_T_Base_AprilTag")
    if raw_T_B_Tag is None:
        raise RuntimeError(
            f"點位 {args.point_id or '(預設)'} 還沒有 tag 快照，"
            "請先在看得到 tag 的位置執行 snapshot_tag。"
        )
    T_B_Tag = np.asarray(raw_T_B_Tag, dtype=np.float64)
    if T_B_Tag.shape != (4, 4):
        raise RuntimeError("tag_snapshot_T_Base_AprilTag 格式錯誤")

    rclpy.init()
    node = make_node(
        args.tag_size if float(args.tag_size) > 0 else 0.0475,
        args.tag_id if int(args.tag_id) >= 0 else 0,
    )
    try:
        wait_feedback(node, need_camera=False)
        T_B_G = pose_to_T(current_tool_pose(node))
        T_Tag_Grab = invert_T(T_B_Tag) @ T_B_G

        T_Tag_Approach = _save_point_teach(data, args.point_id, T_Tag_Grab)

        print("\n========================================")
        print(f"點位 {args.point_id or '(預設)'} 示教完成（用快照換算）")
        print("========================================")
        print("T_Tag_GrabPoint:")
        print(np.array2string(T_Tag_Grab, precision=8, separator=", "))
        print("T_Tag_ApproachPoint (自動退後 100mm 算出):")
        print(np.array2string(T_Tag_Approach, precision=8, separator=", "))
        print("saved:", config_file_for(args.point_id))
        print("========================================")
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


def _resolve_point_params(args):
    data = load_config(args.point_id)
    if "T_Tag_GrabPoint" not in data:
        raise RuntimeError(f"點位 {args.point_id or '(預設)'} 尚未示教。")
    if int(args.tag_id) < 0:
        args.tag_id = int(data["tag_id"])
    if float(args.tag_size) <= 0.0:
        args.tag_size = float(data["tag_size_m"])
    return data


def run_pick(args):
    """單一點位的完整抓取流程：偵測 tag -> 預備點(PTP_T) -> 開爪 ->
    抓取點(LINE_T) -> 閉爪 -> 退出(LINE_T) -> 回主機前方(PTP_T)。"""
    data = _resolve_point_params(args)
    T_G_C = _load_eye(args)

    T_Tag_Grab = np.asarray(data["T_Tag_GrabPoint"], dtype=np.float64)
    T_Tag_Approach = np.asarray(data["T_Tag_ApproachPoint"], dtype=np.float64)

    rclpy.init()
    node = make_node(args.tag_size, args.tag_id)
    try:
        T_B_Tag_now, _, _ = detect_base_tag(node, T_G_C, args.samples)

        T_B_Approach_now = T_B_Tag_now @ T_Tag_Approach
        T_B_Grab_now = T_B_Tag_now @ T_Tag_Grab
        approach_pose = T_to_pose(T_B_Approach_now)
        grab_pose = T_to_pose(T_B_Grab_now)

        print(
            f"\n目前 tag 偵測換算：點位 {args.point_id or '(預設)'} "
            f"預備點={np.array2string(approach_pose, precision=6)} "
            f"抓取點={np.array2string(grab_pose, precision=6)}"
        )

        send_pose(
            node, approach_pose, TRANSIT_VELOCITY,
            f"先移動到點位 {args.point_id or '(預設)'} 預備位置",
            motion="PTP_T", timeout=60.0,
        )

        send_gripper(node, close=False)
        time.sleep(GRIPPER_ACTUATION_DELAY_S)

        send_pose(
            node, grab_pose, PRECISION_VELOCITY,
            f"移動到點位 {args.point_id or '(預設)'} 抓取座標",
            motion="LINE_T", timeout=30.0,
        )

        print(f">>> 已抵達點位 {args.point_id or '(預設)'}，夾爪抓取中... <<<")
        send_gripper(node, close=True)
        time.sleep(GRIPPER_ACTUATION_DELAY_S)

        retreat_pose = grab_pose.copy()
        retreat_pose[0] -= RETREAT_DISTANCE_M
        send_pose(
            node, retreat_pose, PRECISION_VELOCITY,
            f"抓取完成，沿 -X 後退 {RETREAT_DISTANCE_M * 1000:.0f}mm",
            motion="LINE_T", timeout=30.0,
        )

        send_pose(
            node, np.asarray(HOST_FRONT_POSE_M, dtype=np.float64),
            TRANSIT_VELOCITY,
            "返回主機前方",
            motion="PTP_T", timeout=60.0,
        )

        print(f">>> 點位 {args.point_id or '(預設)'} 完成，已回到主機前方！ <<<")
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


def run_pick_all(args):
    raw_ids = [p.strip() for p in str(args.point_ids).split(",") if p.strip()]
    if not raw_ids:
        raise RuntimeError("--point-ids 不能是空的，例如 --point-ids 1,2,3")

    for point_id in raw_ids:
        print("\n========================================")
        print(f"開始執行點位 {point_id}")
        print("========================================")
        args.point_id = point_id
        args.tag_id = -1
        args.tag_size = 0.0
        run_pick(args)
        print(f"點位 {point_id} 完成。")

    print("\n========================================")
    print(f"全部 {len(raw_ids)} 個點位依序完成：{', '.join(raw_ids)}")
    print("========================================")


def show(point_id):
    import yaml
    data = load_config(point_id)
    if not data:
        print(f"尚無點位 {point_id or '(預設)'} 的示教資料：", config_file_for(point_id))
        return
    print(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=[
            "snapshot_tag",
            "teach_point",
            "teach_point_from_snapshot",
            "run_pick",
            "run_pick_all",
            "show",
        ],
    )
    parser.add_argument("--eye-yaml", default="")
    parser.add_argument("--eye-key", default="T_G_C")
    parser.add_argument("--eye-invert", action="store_true")
    parser.add_argument(
        "--tag-id",
        type=int,
        default=0,
        help="沿用小張 tag（tag_id=0，tag_size=0.0475m）。如果之後這顆 tag "
        "可能跟 precision_cable_grip 孔洞旁的 tag 同時入鏡，兩者都是 id=0 "
        "會分不清楚，偵測到的會是不確定哪一顆的結果。",
    )
    parser.add_argument("--tag-size", type=float, default=0.0475)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--point-id", default="")
    parser.add_argument(
        "--point-ids",
        default="",
        help="run_pick_all 專用：逗號分隔的點位清單，例如 1,2,3，會依序執行。",
    )
    args = parser.parse_args()

    if args.mode == "show":
        show(args.point_id)
        return

    if args.mode in {
        "snapshot_tag", "teach_point", "run_pick", "run_pick_all",
    } and not args.eye_yaml:
        raise RuntimeError("此功能需要 Eye-in-Hand calibration YAML")

    if args.mode == "snapshot_tag":
        snapshot_tag(args)
    elif args.mode == "teach_point":
        teach_point(args)
    elif args.mode == "teach_point_from_snapshot":
        teach_point_from_snapshot(args)
    elif args.mode == "run_pick":
        run_pick(args)
    elif args.mode == "run_pick_all":
        run_pick_all(args)


if __name__ == "__main__":
    main()
