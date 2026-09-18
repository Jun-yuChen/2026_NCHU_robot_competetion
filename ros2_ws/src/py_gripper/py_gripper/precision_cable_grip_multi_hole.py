#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
precision_cable_grip.py 的多孔位版本——完全獨立的新檔案，不改動原本的
precision_cable_grip.py。差異只有：

- 每個孔位的示教資料/執行紀錄各自存成獨立檔案（依 --hole-id 命名），
  教下一個孔不會覆蓋掉前一個孔的資料。
- 新增 run_grip_all 模式，用 --hole-ids 1,2,3 這種逗號列表，
  依序對每個孔位跑一次完整的 4 步驟抓取流程。

其餘邏輯（4 步驟流程、示教方式、夾爪控制）跟 precision_cable_grip.py
完全相同，一樣依賴 apriltag_three_view_teach.py。
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml

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


def config_file_for(hole_id):
    suffix = f"_{hole_id}" if hole_id else ""
    return DATA_DIR / f"precision_cable_grip_teach{suffix}.yaml"


def runtime_file_for(hole_id):
    suffix = f"_{hole_id}" if hole_id else ""
    return DATA_DIR / f"precision_cable_grip_runtime{suffix}.yaml"


def load_config(hole_id=""):
    config_file = config_file_for(hole_id)
    if not config_file.is_file():
        return {}
    with config_file.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def save_config(data, hole_id=""):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with config_file_for(hole_id).open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
            allow_unicode=True,
        )


def load_runtime_state(hole_id=""):
    runtime_file = runtime_file_for(hole_id)
    if not runtime_file.is_file():
        return {}
    with runtime_file.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def save_runtime_state(data, hole_id=""):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with runtime_file_for(hole_id).open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
            allow_unicode=True,
        )


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
    """Detect the hole-side AprilTag once (stable multi-frame estimate)."""
    wait_feedback(node, need_camera=True)
    T_C_Tag = node.detect_tag_stable(
        count=int(samples),
        timeout=max(6.0, 1.5 * int(samples)),
    )
    T_B_G = pose_to_T(current_tool_pose(node))
    T_B_Tag = T_B_G @ T_G_C @ T_C_Tag
    return T_B_Tag, T_C_Tag, T_B_G


def send_pose(node, pose, velocity, label, motion="LINE_T", timeout=30.0):
    pose = np.asarray(pose, dtype=np.float64).reshape(6)
    print("\n----------------------------------------")
    print(label)
    print("----------------------------------------")
    print(np.array2string(
        pose,
        precision=8,
        separator=", ",
    ))
    node.send_pose(
        pose.tolist(),
        velocity=float(velocity),
        acc_time=0.5,
        motion=motion,
    )
    node.wait_arrived(pose, timeout=float(timeout))
    node.settle(0.35)


def send_gripper(node, close):
    label = "CLOSE" if close else "OPEN"
    state = 1.0 if close else 0.0

    cli = node.create_client(SetIO, "set_io")
    if not cli.wait_for_service(timeout_sec=5.0):
        raise RuntimeError(
            "找不到 /set_io service；請確認 TM Driver / Listen 已連線。"
        )

    if not hasattr(SetIO.Request, "MODULE_ENDEFFECTOR"):
        raise RuntimeError("tm_msgs/SetIO 缺少 MODULE_ENDEFFECTOR")
    if not hasattr(SetIO.Request, "TYPE_DIGITAL_OUT"):
        raise RuntimeError("tm_msgs/SetIO 缺少 TYPE_DIGITAL_OUT")

    req = SetIO.Request()
    req.module = SetIO.Request.MODULE_ENDEFFECTOR
    req.type = SetIO.Request.TYPE_DIGITAL_OUT
    req.pin = 0
    req.state = float(state)

    print(
        f"\nGRIPPER {label}: "
        f"ENDEFFECTOR DO_0 state={state:.1f}"
    )

    future = cli.call_async(req)
    rclpy.spin_until_future_complete(
        node,
        future,
        timeout_sec=5.0,
    )

    if not future.done() or future.result() is None:
        raise RuntimeError(f"夾爪 {label} /set_io 呼叫失敗")

    result = future.result()
    if hasattr(result, "ok") and not bool(result.ok):
        raise RuntimeError(
            f"夾爪 {label} /set_io rejected: {result}"
        )


def snapshot_tag(args):
    """Detect the AprilTag from wherever the arm currently is (must be a
    pose where the tag IS visible — does not have to be the grip point)
    and save T_Base_AprilTag for later reuse by teach_end_from_snapshot.

    用在孔洞旁 tag 在真正夾取終點會被擋住/看不到的情況：先在看得到 tag
    的地方(不用是夾取終點)拍一次記住 tag 現在在哪，再搭配
    teach_end_from_snapshot 在真正的終點只讀姿態、不需要相機。"""
    T_G_C = load_matrix(
        args.eye_yaml,
        args.eye_key,
        invert=args.eye_invert,
    )

    rclpy.init()
    node = make_node(args.tag_size, args.tag_id)
    try:
        T_B_Tag, T_C_Tag, _ = detect_base_tag(
            node,
            T_G_C,
            args.samples,
        )

        data = load_config(args.hole_id)
        data.update({
            "hole_id": args.hole_id,
            "tag_id": int(args.tag_id),
            "tag_size_m": float(args.tag_size),
            "tag_snapshot_saved_at_unix": float(time.time()),
            "tag_snapshot_T_Base_AprilTag": T_B_Tag.tolist(),
            "tag_snapshot_T_Camera_AprilTag": T_C_Tag.tolist(),
        })
        save_config(data, args.hole_id)

        print("\n========================================")
        print(f"Tag 快照已記錄（孔位: {args.hole_id or '(預設)'}）")
        print("========================================")
        print("T_Base_AprilTag:")
        print(np.array2string(
            T_B_Tag,
            precision=8,
            separator=", ",
        ))
        print("接下來把手臂移到真正的夾取終點(即使那裡看不到 tag)，")
        print("執行 teach_end_from_snapshot。")
        print("saved:", config_file_for(args.hole_id))
        print("========================================")
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


def teach_end_from_snapshot(args):
    """Teach the grip endpoint using a previously captured tag snapshot
    instead of a live detection at the grip point itself. Run
    snapshot_tag first from a pose where the tag IS visible."""
    data = load_config(args.hole_id)
    raw_T_B_Tag = data.get("tag_snapshot_T_Base_AprilTag")
    if raw_T_B_Tag is None:
        raise RuntimeError(
            f"孔位 {args.hole_id or '(預設)'} 還沒有 tag 快照，"
            "請先在看得到 tag 的位置執行 snapshot_tag。"
        )

    T_B_Tag = np.asarray(raw_T_B_Tag, dtype=np.float64)
    if T_B_Tag.shape != (4, 4):
        raise RuntimeError("tag_snapshot_T_Base_AprilTag 格式錯誤")

    rclpy.init()
    node = make_node(
        args.tag_size if float(args.tag_size) > 0 else 0.130,
        args.tag_id if int(args.tag_id) >= 0 else 0,
    )
    try:
        wait_feedback(node, need_camera=False)
        T_B_G = pose_to_T(current_tool_pose(node))

        T_Tag_ToolEnd = invert_T(T_B_Tag) @ T_B_G
        end_pose = T_to_pose(T_B_G)

        start_pose = data.get("teach_start_tool_pose_xyz_rpy_rad")
        dx = None
        if isinstance(start_pose, list) and len(start_pose) == 6:
            dx = float(end_pose[0] - float(start_pose[0]))

        data.update({
            "hole_id": args.hole_id,
            "teach_end_saved_at_unix": float(time.time()),
            "teach_grip_end_tool_pose_xyz_rpy_rad": [
                float(v) for v in end_pose
            ],
            "teach_end_T_Base_AprilTag": T_B_Tag.tolist(),
            "T_AprilTag_ToolEnd": T_Tag_ToolEnd.tolist(),
            "grip_x_delta_m_reference_only": dx,
            "teach_end_source": (
                "snapshot：終點本身看不到 tag，用之前 snapshot_tag "
                "記錄的 tag 位置換算，不是即時偵測。"
            ),
            "runtime_end_alignment": (
                "After reaching runtime start, detect AprilTag again, "
                "then T_Base_ToolEnd = T_Base_Tag_now @ T_AprilTag_ToolEnd"
            ),
        })
        save_config(data, args.hole_id)

        print("\n========================================")
        print(f"精準夾取終點示教完成（孔位: {args.hole_id or '(預設)'}，用快照換算）")
        print("========================================")
        print("T_AprilTag_ToolEnd:")
        print(np.array2string(
            T_Tag_ToolEnd,
            precision=8,
            separator=", ",
        ))
        if dx is not None:
            print(f"teach Base-X delta (reference only) = {dx:.8f} m")
        print("saved:", config_file_for(args.hole_id))
        print("========================================")
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


def derive_start_from_end(args):
    """Compute T_AprilTag_ToolStart from the already-taught end point by
    retreating along base-frame X, without a fresh physical teach step or
    live tag detection. Requires teach_end to have been run first."""
    data = load_config(args.hole_id)
    raw_T_Tag_End = data.get("T_AprilTag_ToolEnd")
    raw_T_B_Tag_ref = data.get("teach_end_T_Base_AprilTag")
    if raw_T_Tag_End is None or raw_T_B_Tag_ref is None:
        raise RuntimeError(
            f"孔位 {args.hole_id or '(預設)'} 尚未示教終點 (teach_end)，"
            "無法反推起點。"
        )

    T_Tag_ToolEnd = np.asarray(raw_T_Tag_End, dtype=np.float64)
    T_B_Tag_ref = np.asarray(raw_T_B_Tag_ref, dtype=np.float64)
    if T_Tag_ToolEnd.shape != (4, 4) or T_B_Tag_ref.shape != (4, 4):
        raise RuntimeError("teach_end 資料格式錯誤")

    retreat = float(args.retreat_distance)
    if not np.isfinite(retreat) or retreat < 0.0:
        raise RuntimeError("--retreat-distance 必須 >= 0")

    # 終點在 base 座標系下的姿態（用示教當下偵測到的 tag 位置換算）
    T_B_ToolEnd_ref = T_B_Tag_ref @ T_Tag_ToolEnd

    # 沿 base X 退後，姿態(旋轉)維持跟終點完全一樣
    T_B_ToolStart_ref = T_B_ToolEnd_ref.copy()
    T_B_ToolStart_ref[0, 3] -= retreat

    # 重新換算回「相對 tag」的表示方式，執行時才能用即時偵測到的 tag
    # 位置重新算出當下的起點
    T_Tag_ToolStart = invert_T(T_B_Tag_ref) @ T_B_ToolStart_ref
    start_pose = T_to_pose(T_B_ToolStart_ref)

    data.update({
        "hole_id": args.hole_id,
        "teach_start_saved_at_unix": float(time.time()),
        "teach_start_tool_pose_xyz_rpy_rad": [
            float(v) for v in start_pose
        ],
        "teach_T_Base_AprilTag": T_B_Tag_ref.tolist(),
        "T_AprilTag_ToolStart": T_Tag_ToolStart.tolist(),
        "start_derived_from_end_retreat_m": retreat,
        "motion_definition": {
            "runtime_alignment": (
                "Detect current hole-side AprilTag, then "
                "T_Base_ToolStart = T_Base_Tag_now @ T_AprilTag_ToolStart"
            ),
            "final_grip_motion": (
                "Base X only; Y/Z/Rx/Ry/Rz remain at runtime aligned start"
            ),
        },
    })
    save_config(data, args.hole_id)

    print("\n========================================")
    print(
        f"起點已從終點反推完成（孔位: {args.hole_id or '(預設)'}，"
        f"退後 {retreat * 1000:.0f}mm）"
    )
    print("========================================")
    print("T_AprilTag_ToolStart:")
    print(np.array2string(
        T_Tag_ToolStart,
        precision=8,
        separator=", ",
    ))
    print("saved:", config_file_for(args.hole_id))
    print("========================================")


def teach_start(args):
    """Teach the precise pre-grip point relative to the hole-side AprilTag."""
    T_G_C = load_matrix(
        args.eye_yaml,
        args.eye_key,
        invert=args.eye_invert,
    )

    rclpy.init()
    node = make_node(args.tag_size, args.tag_id)
    try:
        T_B_Tag, T_C_Tag, T_B_G = detect_base_tag(
            node,
            T_G_C,
            args.samples,
        )
        T_Tag_ToolStart = invert_T(T_B_Tag) @ T_B_G
        start_pose = T_to_pose(T_B_G)

        data = load_config(args.hole_id)
        data.update({
            "hole_id": args.hole_id,
            "tag_id": int(args.tag_id),
            "tag_size_m": float(args.tag_size),
            "teach_start_saved_at_unix": float(time.time()),
            "teach_start_tool_pose_xyz_rpy_rad": [
                float(v) for v in start_pose
            ],
            "teach_T_Base_AprilTag": T_B_Tag.tolist(),
            "teach_T_Camera_AprilTag": T_C_Tag.tolist(),
            "T_AprilTag_ToolStart": T_Tag_ToolStart.tolist(),
            "motion_definition": {
                "runtime_alignment": (
                    "Detect current hole-side AprilTag, then "
                    "T_Base_ToolStart = T_Base_Tag_now @ T_AprilTag_ToolStart"
                ),
                "final_grip_motion": (
                    "Base X only; Y/Z/Rx/Ry/Rz remain at runtime aligned start"
                ),
            },
        })
        save_config(data, args.hole_id)

        print("\n========================================")
        print(f"精準夾取起點示教完成（孔位: {args.hole_id or '(預設)'}）")
        print("========================================")
        print("已同時記錄孔位旁 AprilTag 與目前精準夾取起點。")
        print("T_AprilTag_ToolStart:")
        print(np.array2string(
            T_Tag_ToolStart,
            precision=8,
            separator=", ",
        ))
        print("saved:", config_file_for(args.hole_id))
        print("========================================")
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


def teach_end(args):
    """Teach exact grip endpoint relative to the same hole-side AprilTag.

    可以獨立示教，不要求已經先教過起點——本檔案的建議流程是先教終點，
    再用 derive_start_from_end 從終點反推起點。"""
    data = load_config(args.hole_id)

    T_G_C = load_matrix(
        args.eye_yaml,
        args.eye_key,
        invert=args.eye_invert,
    )

    rclpy.init()
    node = make_node(args.tag_size, args.tag_id)
    try:
        T_B_Tag, T_C_Tag, T_B_G = detect_base_tag(
            node,
            T_G_C,
            args.samples,
        )

        T_Tag_ToolEnd = invert_T(T_B_Tag) @ T_B_G
        end_pose = T_to_pose(T_B_G)

        start_pose = data.get("teach_start_tool_pose_xyz_rpy_rad")
        dx = None
        if isinstance(start_pose, list) and len(start_pose) == 6:
            dx = float(end_pose[0] - float(start_pose[0]))

        data.update({
            "hole_id": args.hole_id,
            "tag_id": int(args.tag_id),
            "tag_size_m": float(args.tag_size),
            "teach_end_saved_at_unix": float(time.time()),
            "teach_grip_end_tool_pose_xyz_rpy_rad": [
                float(v) for v in end_pose
            ],
            "teach_end_T_Base_AprilTag": T_B_Tag.tolist(),
            "teach_end_T_Camera_AprilTag": T_C_Tag.tolist(),
            "T_AprilTag_ToolEnd": T_Tag_ToolEnd.tolist(),
            "grip_x_delta_m_reference_only": dx,
            "runtime_end_alignment": (
                "After reaching runtime start, detect AprilTag again, "
                "then T_Base_ToolEnd = T_Base_Tag_now @ T_AprilTag_ToolEnd"
            ),
        })
        save_config(data, args.hole_id)

        print("\n========================================")
        print(f"精準夾取終點示教完成（孔位: {args.hole_id or '(預設)'}）")
        print("========================================")
        print("已重新偵測孔位旁 AprilTag，並記錄終點相對 Tag 的完整 6D 關係。")
        print("T_AprilTag_ToolEnd:")
        print(np.array2string(
            T_Tag_ToolEnd,
            precision=8,
            separator=", ",
        ))
        if dx is not None:
            print(f"teach Base-X delta (reference only) = {dx:.8f} m")
        print("saved:", config_file_for(args.hole_id))
        print("========================================")
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


def compute_runtime_end(node, args, T_G_C):
    data = load_config(args.hole_id)
    raw = data.get("T_AprilTag_ToolEnd")
    if raw is None:
        raise RuntimeError(
            f"孔位 {args.hole_id or '(預設)'} 尚未示教 T_AprilTag_ToolEnd"
        )

    T_Tag_ToolEnd = np.asarray(
        raw,
        dtype=np.float64,
    )
    if T_Tag_ToolEnd.shape != (4, 4):
        raise RuntimeError("T_AprilTag_ToolEnd 格式錯誤")

    T_B_Tag_now, _, _ = detect_base_tag(
        node,
        T_G_C,
        args.samples,
    )
    T_B_ToolEnd_now = T_B_Tag_now @ T_Tag_ToolEnd
    return T_to_pose(T_B_ToolEnd_now), T_B_Tag_now


def compute_runtime_start(node, args, T_G_C):
    data = load_config(args.hole_id)
    T_Tag_ToolStart_raw = data.get("T_AprilTag_ToolStart")
    if T_Tag_ToolStart_raw is None:
        raise RuntimeError(
            f"孔位 {args.hole_id or '(預設)'} 尚未示教 T_AprilTag_ToolStart"
        )

    T_Tag_ToolStart = np.asarray(
        T_Tag_ToolStart_raw,
        dtype=np.float64,
    )
    if T_Tag_ToolStart.shape != (4, 4):
        raise RuntimeError("T_AprilTag_ToolStart 格式錯誤")

    T_B_Tag_now, _, _ = detect_base_tag(
        node,
        T_G_C,
        args.samples,
    )
    T_B_ToolStart_now = T_B_Tag_now @ T_Tag_ToolStart
    return T_to_pose(T_B_ToolStart_now), T_B_Tag_now


def preview(args):
    T_G_C = load_matrix(
        args.eye_yaml,
        args.eye_key,
        invert=args.eye_invert,
    )
    rclpy.init()
    node = make_node(args.tag_size, args.tag_id)
    try:
        pose, T_B_Tag_now = compute_runtime_start(
            node,
            args,
            T_G_C,
        )
        print("\n========================================")
        print(f"目前 AprilTag → 精準夾取起點預覽（孔位: {args.hole_id or '(預設)'}）")
        print("========================================")
        print("T_Base_AprilTag now:")
        print(np.array2string(
            T_B_Tag_now,
            precision=8,
            separator=", ",
        ))
        print("runtime start pose:")
        print(np.array2string(
            np.asarray(pose),
            precision=8,
            separator=", ",
        ))
        print("========================================")
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


def move_to_start(args):
    T_G_C = load_matrix(
        args.eye_yaml,
        args.eye_key,
        invert=args.eye_invert,
    )
    rclpy.init()
    node = make_node(args.tag_size, args.tag_id)
    try:
        pose, _ = compute_runtime_start(
            node,
            args,
            T_G_C,
        )
        send_pose(
            node,
            pose,
            args.transit_velocity,
            "偵測孔位 AprilTag 後移動到示教精準夾取起點",
            motion="PTP_T",
            timeout=60.0,
        )
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


def _resolve_grip_params(args):
    data = load_config(args.hole_id)
    if "T_AprilTag_ToolEnd" not in data:
        raise RuntimeError(
            f"孔位 {args.hole_id or '(預設)'} 尚未示教夾取終點 AprilTag 關係。"
        )
    if int(args.tag_id) < 0:
        args.tag_id = int(data["tag_id"])
    if float(args.tag_size) <= 0.0:
        args.tag_size = float(data["tag_size_m"])


def _load_eye(args):
    return load_matrix(
        args.eye_yaml,
        args.eye_key,
        invert=args.eye_invert,
    )


def grip_step1(args):
    """Tag #1 -> move to precise grip start."""
    _resolve_grip_params(args)
    T_G_C = _load_eye(args)
    rclpy.init()
    node = make_node(args.tag_size, args.tag_id)
    try:
        send_gripper(node, close=False)
        node.settle(0.25)

        start_pose, T_B_Tag = compute_runtime_start(
            node, args, T_G_C
        )
        send_pose(
            node,
            start_pose,
            args.transit_velocity,
            f"Step 1 / Tag #1 → 精準夾取起點（孔位: {args.hole_id or '(預設)'}）",
            motion="PTP_T",
            timeout=60.0,
        )
        save_runtime_state({
            "step": 1,
            "hole_id": args.hole_id,
            "tag1_T_Base_AprilTag": T_B_Tag.tolist(),
            "runtime_start_pose": [
                float(v) for v in start_pose
            ],
        }, args.hole_id)
        print("Step 1 完成：已到精準夾取起點。")
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


def grip_step2(args):
    """Tag #2 -> compute endpoint -> move Y only to endpoint Y."""
    _resolve_grip_params(args)
    T_G_C = _load_eye(args)
    rclpy.init()
    node = make_node(args.tag_size, args.tag_id)
    try:
        end_pose, T_B_Tag = compute_runtime_end(
            node, args, T_G_C
        )
        wait_feedback(node, need_camera=False)
        current = current_tool_pose(node)
        y_pose = current.copy()
        y_pose[1] = float(end_pose[1])

        send_pose(
            node,
            y_pose,
            args.align_velocity,
            f"Step 2 / Tag #2 → 只移動 Base Y 到夾取終點 Y（孔位: {args.hole_id or '(預設)'}）",
        )

        state = load_runtime_state(args.hole_id)
        state.update({
            "step": 2,
            "hole_id": args.hole_id,
            "tag2_T_Base_AprilTag": T_B_Tag.tolist(),
            "tag2_predicted_end_pose": [
                float(v) for v in end_pose
            ],
            "y_aligned_pose": [
                float(v) for v in y_pose
            ],
        })
        save_runtime_state(state, args.hole_id)
        print("Step 2 完成：Y 已對齊夾取終點 Y。")
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


def grip_step3(args):
    """Tag #3 -> compute and SAVE final endpoint. No motion after detection."""
    _resolve_grip_params(args)
    T_G_C = _load_eye(args)
    rclpy.init()
    node = make_node(args.tag_size, args.tag_id)
    try:
        end_pose, T_B_Tag = compute_runtime_end(
            node, args, T_G_C
        )
        state = load_runtime_state(args.hole_id)
        state.update({
            "step": 3,
            "hole_id": args.hole_id,
            "tag3_T_Base_AprilTag": T_B_Tag.tolist(),
            "tag3_final_end_pose": [
                float(v) for v in end_pose
            ],
        })
        save_runtime_state(state, args.hole_id)

        print("\n========================================")
        print(f"Step 3 / Tag #3 完成（孔位: {args.hole_id or '(預設)'}）")
        print("已鎖定最終夾取終點；接下來不再掃 AprilTag。")
        print("final endpoint:")
        print(np.array2string(
            np.asarray(end_pose),
            precision=8,
            separator=", ",
        ))
        print("========================================")
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


def grip_step4(args):
    """No AprilTag. X-only -> verify -> CLOSE -> pull out."""
    state = load_runtime_state(args.hole_id)
    raw = state.get("tag3_final_end_pose")
    if not isinstance(raw, list) or len(raw) != 6:
        raise RuntimeError(
            f"孔位 {args.hole_id or '(預設)'} 尚未完成 Step 3 / Tag #3，"
            "沒有鎖定的最終夾取終點。"
        )

    pull_out = float(args.pull_out_distance)
    if not np.isfinite(pull_out) or pull_out < 0.0:
        raise RuntimeError("夾取後拔出距離必須 >= 0")

    rclpy.init()
    node = make_node(
        args.tag_size if float(args.tag_size) > 0 else 0.130,
        args.tag_id if int(args.tag_id) >= 0 else 0,
    )
    try:
        wait_feedback(node, need_camera=False)
        current = current_tool_pose(node)
        final_ref = np.asarray(raw, dtype=np.float64)

        x_pose = current.copy()
        x_pose[0] = float(final_ref[0])

        approach_dx = float(x_pose[0] - current[0])

        send_pose(
            node,
            x_pose,
            args.grip_velocity,
            f"Step 4 / 不再掃 Tag → Base X-only 慢速前進到終點（孔位: {args.hole_id or '(預設)'}）",
        )

        node.settle(0.50)
        wait_feedback(node, need_camera=False)
        reached = current_tool_pose(node)
        x_error = abs(float(reached[0] - x_pose[0]))

        print(
            f"夾取終點 X feedback error = "
            f"{x_error * 1000.0:.3f} mm"
        )
        if x_error > 0.002:
            raise RuntimeError(
                "X 尚未穩定到終點（誤差 > 2 mm），"
                "本次不執行 CLOSE。"
            )

        send_gripper(node, close=True)
        node.settle(float(args.grip_wait))

        if pull_out > 1e-12:
            if abs(approach_dx) <= 1e-9:
                raise RuntimeError(
                    "X 前進距離太小，無法判斷拔出方向。"
                )
            direction = 1.0 if approach_dx > 0.0 else -1.0
            pull_pose = x_pose.copy()
            pull_pose[0] -= direction * pull_out
            send_pose(
                node,
                pull_pose,
                args.grip_velocity,
                "CLOSE 後沿 Base X 反方向拔出",
            )

        state["step"] = 4
        state["completed"] = True
        save_runtime_state(state, args.hole_id)
        print("Step 4 完成：X-only 到位 → CLOSE → 拔出。")
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


def run_grip(args):
    """One-click version of the exact same 4-step flow for one hole."""
    grip_step1(args)
    grip_step2(args)
    grip_step3(args)
    grip_step4(args)


def run_grip_all(args):
    """Run the full 4-step grip flow for each hole in --hole-ids, in order."""
    raw_ids = [h.strip() for h in str(args.hole_ids).split(",") if h.strip()]
    if not raw_ids:
        raise RuntimeError(
            "--hole-ids 不能是空的，例如 --hole-ids 1,2,3"
        )

    for hole_id in raw_ids:
        print("\n========================================")
        print(f"開始執行孔位 {hole_id}")
        print("========================================")
        args.hole_id = hole_id
        args.tag_id = -1
        args.tag_size = 0.0
        run_grip(args)
        print(f"孔位 {hole_id} 完成。")

    print("\n========================================")
    print(f"全部 {len(raw_ids)} 個孔位依序完成：{', '.join(raw_ids)}")
    print("========================================")


def test_gripper(close, args):
    rclpy.init()
    node = make_node(args.tag_size, args.tag_id)
    try:
        send_gripper(node, close=bool(close))
        node.settle(0.3)
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


def show(hole_id):
    data = load_config(hole_id)
    if not data:
        print(f"尚無孔位 {hole_id or '(預設)'} 的精準夾取示教資料：", config_file_for(hole_id))
        return
    print(yaml.safe_dump(
        data,
        sort_keys=False,
        allow_unicode=True,
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=[
            "teach_start",
            "teach_end",
            "snapshot_tag",
            "teach_end_from_snapshot",
            "derive_start_from_end",
            "preview",
            "move_to_start",
            "grip_step1",
            "grip_step2",
            "grip_step3",
            "grip_step4",
            "run_grip",
            "run_grip_all",
            "gripper_open",
            "gripper_close",
            "show",
        ],
    )
    parser.add_argument("--eye-yaml", default="")
    parser.add_argument("--eye-key", default="T_G_C")
    parser.add_argument("--eye-invert", action="store_true")
    parser.add_argument("--tag-id", type=int, default=0)
    parser.add_argument("--tag-size", type=float, default=0.130)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument(
        "--align-velocity",
        type=float,
        default=0.15,
        help="tm_driver 內部用 vel/pi 無條件捨去成整數百分比，"
        "0.03 會直接捨去成 0%（完全不會動），至少要 >0.0315。",
    )
    parser.add_argument(
        "--transit-velocity",
        type=float,
        default=0.6,
        help="飛到起點(PTP_T 長距離轉場)用的速度，跟 align-velocity 分開，避免用微調速度跑長距離逾時。",
    )
    parser.add_argument(
        "--grip-velocity",
        type=float,
        default=0.05,
        help="Step 4 最終逼近速度。舊預設 0.005 會被 tm_driver 捨去成 0%（完全不會動），"
        "至少要 >0.0315 才會有任何動作。",
    )
    parser.add_argument("--grip-wait", type=float, default=0.5)
    parser.add_argument("--pull-out-distance", type=float, default=0.03)
    parser.add_argument(
        "--retreat-distance",
        type=float,
        default=0.10,
        help="derive_start_from_end 專用：起點相對終點沿 base X 退後的距離(m)，預設沿用 arm_goto_fixed_table.py 的 100mm。",
    )
    parser.add_argument(
        "--hole-id",
        default="",
        help="這個孔位的識別碼，用來把示教/執行資料存成獨立檔案，不給就跟單孔位版本行為一樣。",
    )
    parser.add_argument(
        "--hole-ids",
        default="",
        help="run_grip_all 專用：逗號分隔的孔位清單，例如 1,2,3，會依序執行。",
    )
    args = parser.parse_args()

    if args.mode == "show":
        show(args.hole_id)
        return

    if args.mode in {
        "teach_start",
        "teach_end",
        "snapshot_tag",
        "preview",
        "move_to_start",
        "grip_step1",
        "grip_step2",
        "grip_step3",
        "run_grip",
        "run_grip_all",
    } and not args.eye_yaml:
        raise RuntimeError(
            "此功能需要 Eye-in-Hand calibration YAML"
        )

    if args.mode == "teach_start":
        teach_start(args)
    elif args.mode == "teach_end":
        teach_end(args)
    elif args.mode == "snapshot_tag":
        snapshot_tag(args)
    elif args.mode == "teach_end_from_snapshot":
        teach_end_from_snapshot(args)
    elif args.mode == "derive_start_from_end":
        derive_start_from_end(args)
    elif args.mode == "preview":
        preview(args)
    elif args.mode == "move_to_start":
        move_to_start(args)
    elif args.mode == "grip_step1":
        grip_step1(args)
    elif args.mode == "grip_step2":
        grip_step2(args)
    elif args.mode == "grip_step3":
        grip_step3(args)
    elif args.mode == "grip_step4":
        grip_step4(args)
    elif args.mode == "run_grip":
        run_grip(args)
    elif args.mode == "run_grip_all":
        run_grip_all(args)
    elif args.mode == "gripper_open":
        test_gripper(False, args)
    elif args.mode == "gripper_close":
        test_gripper(True, args)


if __name__ == "__main__":
    main()
